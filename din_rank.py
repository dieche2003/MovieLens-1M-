r"""第五步：DIN 精排及 BGE 语义消融。

只读取已有召回候选和特征，不新增召回、特征工程、负采样或重排。
运行：python D:\xiangmu-dianying\din_rank.py
输出 data/din_score_valid.pkl、reports/din_report.json、figures/din_auc_curve.png。
"""
from pathlib import Path
from collections import defaultdict
import copy, json, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(r"D:\xiangmu-dianying"); DATA=ROOT/"data"; REPORTS=ROOT/"reports"; FIGURES=ROOT/"figures"
SEED=42; MAX_HISTORY=20; BATCH=1024; TRAIN_ROWS=100000; EPOCHS=2
BG=[f"bge_pca_{i:02d}" for i in range(32)]

def seed_all():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

def load_inputs():
    candidates=pd.read_pickle(DATA/"recall_candidates_valid.pkl")
    valid=pd.read_pickle(DATA/"features_valid.pkl")
    train=pd.read_pickle(DATA/"features_train.pkl")
    movies=pd.read_pickle(DATA/"df_movies.pkl")
    print(f"[load] valid={len(valid):,}, users={len(candidates):,}, train_features={len(train):,}",flush=True)
    return candidates,valid,train,movies

def encode_categories(train,valid):
    frames=[train,valid]; maps={}
    for col in ["gender","age","occupation"]:
        vals=sorted(set(str(x) for f in frames for x in f[col].astype(str).unique())); maps[col]={v:i+1 for i,v in enumerate(vals)}
    users=sorted(set(train.user_id.astype(int))|set(valid.user_id.astype(int))); items=sorted(set(train.movie_id.astype(int))|set(valid.movie_id.astype(int)))
    maps["user_id"]={x:i+1 for i,x in enumerate(users)};maps["movie_id"]={x:i+1 for i,x in enumerate(items)}
    return maps

def histories_to_array(values,movie_map):
    out=np.zeros((len(values),MAX_HISTORY),np.int64)
    for i,h in enumerate(values):
        h=[movie_map.get(int(x),0) for x in list(h)[-MAX_HISTORY:]];
        if h: out[i,-len(h):]=h
    return out

class DIN(nn.Module):
    def __init__(self,nu,ni,cat_sizes,use_bge):
        super().__init__(); self.use_bge=use_bge; self.ue=nn.Embedding(nu+1,16,padding_idx=0);self.ie=nn.Embedding(ni+1,16,padding_idx=0);self.he=self.ie
        self.cats=nn.ModuleDict({k:nn.Embedding(v+1,4,padding_idx=0) for k,v in cat_sizes.items()})
        self.att=nn.Sequential(nn.Linear(16*4,32),nn.ReLU(),nn.Linear(32,1))
        in_dim=16+16+4*3+3+(32 if use_bge else 0)+16
        self.mlp=nn.Sequential(nn.Linear(in_dim,96),nn.ReLU(),nn.Dropout(.1),nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,u,i,cats,num,hist,bge):
        ue=self.ue(u); ie=self.ie(i); he=self.he(hist); target=ie.unsqueeze(1).expand_as(he)
        x=torch.cat([he,target,he-target,he*target],2); w=torch.softmax(self.att(x).masked_fill(hist.eq(0).unsqueeze(2),-1e9),1); seq=(w*he).sum(1)
        z=[ue,ie,seq]+[self.cats[k](cats[k]) for k in ("gender","age","occupation")]+[num]
        if self.use_bge:z.append(bge)
        return self.mlp(torch.cat(z,1)).squeeze(1)

def feature_arrays(df,maps):
    cat={c:df[c].astype(str).map(maps[c]).fillna(0).to_numpy(np.int64) for c in ["gender","age","occupation"]}
    nums=df[["movie_popularity","user_activity","user_genre_mean_rating"]].fillna(0).to_numpy(np.float32); hist=histories_to_array(df.history_movie_ids,maps["movie_id"]); bge=df[BG].fillna(0).to_numpy(np.float32); u=df.user_id.astype(int).map(maps["user_id"]).to_numpy(np.int64); i=df.movie_id.astype(int).map(maps["movie_id"]).fillna(0).to_numpy(np.int64)
    return {"u":u,"i":i,"cats":cat,"num":nums,"hist":hist,"bge":bge}

def batches(arr,y,device,shuffle=True):
    n=len(y);order=np.arange(n)
    if shuffle:np.random.default_rng(SEED).shuffle(order)
    for st in range(0,n,BATCH):
        ix=order[st:st+BATCH]; yield ix,{"u":torch.tensor(arr["u"][ix],device=device),"i":torch.tensor(arr["i"][ix],device=device),"cats":{k:torch.tensor(v[ix],device=device) for k,v in arr["cats"].items()},"num":torch.tensor(arr["num"][ix],device=device),"hist":torch.tensor(arr["hist"][ix],device=device),"bge":torch.tensor(arr["bge"][ix],device=device)},torch.tensor(y[ix],dtype=torch.float32,device=device)

def train_model(train,maps,use_bge,device):
    n=min(TRAIN_ROWS,len(train)); idx=np.linspace(0,len(train)-1,n,dtype=int); sample=train.iloc[idx].reset_index(drop=True); arr=feature_arrays(sample,maps); y=sample.label.to_numpy(np.float32); cat_sizes={k:len(v) for k,v in maps.items() if k in ["gender","age","occupation"]}; model=DIN(len(maps["user_id"]),len(maps["movie_id"]),cat_sizes,use_bge).to(device);opt=torch.optim.Adam(model.parameters(),lr=2e-3);lossfn=nn.BCEWithLogitsLoss(); curves=[]
    for ep in range(EPOCHS):
        model.train();total=0
        for ix,x,t in batches(arr,y,device):
            opt.zero_grad();loss=lossfn(model(**x),t);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
        curves.append(total/n);print(f"[DIN {'BGE' if use_bge else 'base'}] device={device}, epoch={ep+1}/{EPOCHS}, rows={n:,}, loss={total/n:.5f}",flush=True)
    return model,curves

def make_candidates(candidates,valid,movies,observed_train,maps):
    base=valid.set_index("user_id"); movie=movies.set_index("movie_id"); rows=[]; freq=observed_train.movie_id.value_counts(); times={int(m):np.sort(g.timestamp.to_numpy(np.int64)) for m,g in observed_train.groupby("movie_id")}
    for uid,items in candidates.items():
        q=base.loc[int(uid)]; ts=int(q.timestamp); hist=list(q.history_movie_ids); user_activity=int(q.user_activity); genre_mean=float(q.user_genre_mean_rating)
        for mid in items:
            m=int(mid); pop=int(np.searchsorted(times.get(m,np.array([],np.int64)),ts,side="left")); title=str(movie.loc[m].title);genres=str(movie.loc[m].genres); rows.append((uid,m,pop,user_activity,genre_mean,hist,str(q.gender),str(q.age),str(q.occupation),title,genres,int(q.label) if m==int(q.movie_id) else 0))
    d=pd.DataFrame(rows,columns=["user_id","movie_id","movie_popularity","user_activity","user_genre_mean_rating","history_movie_ids","gender","age","occupation","title","genres","label"])
    emb=pd.read_pickle(DATA/"movie_bge_emb.pkl")["emb_map"]; mat=np.stack([np.asarray(emb[int(m)],np.float32) for m in d.movie_id]);d[BG]=mat.tolist();print(f"[candidates] rows={len(d):,}, users={d.user_id.nunique():,}",flush=True);return d

def predict(model,df,maps,device):
    arr=feature_arrays(df,maps);out=[]; model.eval()
    with torch.no_grad():
        for _,x,t in batches(arr,np.zeros(len(df),np.float32),device,False):out.extend(torch.sigmoid(model(**x)).cpu().numpy().tolist())
    return np.asarray(out,np.float32)

def metrics(df,scores,tail):
    y=df.label.to_numpy(int); valid_mask=np.ones(len(df),bool); tail_mask=df.movie_id.isin(tail).to_numpy(); result={}
    def calc(mask):
        yy=y[mask];ss=scores[mask]
        if len(np.unique(yy))<2:return {"AUC":None,"LogLoss":None,"GAUC":None,"rows":int(mask.sum())}
        auc=float(roc_auc_score(yy,ss));ll=float(log_loss(yy,np.clip(ss,1e-6,1-1e-6))); groups=[]
        for _,g in df.loc[mask].assign(_s=ss).groupby("user_id"):
            if g.label.nunique()>1:groups.append((len(g),roc_auc_score(g.label,g._s)))
        gauc=float(sum(n*a for n,a in groups)/sum(n for n,_ in groups)) if groups else None
        return {"AUC":auc,"LogLoss":ll,"GAUC":gauc,"rows":int(mask.sum()),"users_with_both_labels":len(groups)}
    result["overall"]=calc(valid_mask);result["long_tail"]=calc(tail_mask);return result

def main():
    seed_all();REPORTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True);candidates,valid,train,movies=load_inputs();maps=encode_categories(train,valid);observed=train.loc[train.sample_source.ne("random_unobserved")&train.rating.notna(),["user_id","movie_id","rating","timestamp"]];cand=make_candidates(candidates,valid,movies,observed,maps);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(f"[device] {device}",flush=True)
    freq=observed.movie_id.value_counts();tail=set(freq[freq<=freq.quantile(.2)].index.astype(int));report={"device":str(device),"train_rows":min(TRAIN_ROWS,len(train)),"candidate_rows":len(cand),"tail_definition":"training observed movie count <= 20th percentile","tail_items":len(tail),"experiments":{}}
    score_map={}
    for name,use in [("DIN_base",False),("DIN_BGE",True)]:
        model,curve=train_model(train,maps,use,device);scores=predict(model,cand,maps,device);score_map[name]=scores;report["experiments"][name]={"training_loss":curve,"valid":metrics(cand,scores,tail)};print(f"[{name}] AUC={report['experiments'][name]['valid']['overall']['AUC']}",flush=True)
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
    output={}
    for row,s0,s1 in zip(cand.itertuples(index=False),score_map["DIN_base"],score_map["DIN_BGE"]):output.setdefault(int(row.user_id),[]).append((int(row.movie_id),float(s1)))
    for u in output:output[u].sort(key=lambda x:x[1],reverse=True)
    pd.to_pickle(output,DATA/"din_score_valid.pkl");(REPORTS/"din_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(1,2,figsize=(12,4.5));
    for name,v in report["experiments"].items():ax[0].plot(range(1,len(v["training_loss"])+1),v["training_loss"],marker="o",label=name)
    ax[0].set(xlabel="epoch",ylabel="BCE loss",title="DIN training curves");ax[0].legend()
    ks=["AUC","GAUC"];x=np.arange(2);w=.35
    for j,name in enumerate(report["experiments"]):ax[1].bar(x+j*w,[report["experiments"][name]["valid"]["overall"][k] or 0 for k in ks],w,label=name)
    ax[1].set(xticks=x+w/2,xticklabels=ks,ylim=(0,1),title="Valid overall ranking metrics");ax[1].legend();fig.tight_layout();fig.savefig(FIGURES/"din_auc_curve.png",dpi=160);plt.close(fig);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# 消融分析（面试说明）：DIN 基线只使用用户画像、统计特征和历史行为序列，
# 实验组额外加入电影标题 BGE 的 32 维 PCA 向量。整体 AUC/GAUC 反映候选池内排序，
# 长尾子集单独计算，用于判断语义信息是否帮助交互稀疏的电影。若长尾指标提升，
# 可解释为内容信号补充了协同信号；若没有提升，可能是 MovieLens 标题短、PCA 压缩损失、
# 或召回候选池本身未覆盖长尾目标。该离线显式评分实验不能直接等价线上 CTR，且 test
# 必须留到最终验收，不能用 valid 指标调参后再重复报告。
