"""第四步：四路时序安全召回，只产出候选池，不做粗排/精排/重排。

输入：data/df_movies.pkl、data/train.csv、data/features_valid.pkl、data/movie_bge_emb.pkl。
输出：data/recall_candidates_valid.pkl、reports/recall_report.json、
figures/recall_metrics.png。

所有候选均在 valid 查询时间 t 上生成，用户历史只使用 timestamp < t 的训练真实评分。
双塔只抽样训练样本，小批量 CPU/CUDA 训练；test 不读取、不参与评估。
"""
from pathlib import Path
from collections import Counter, defaultdict
import json, math
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\xiangmu-dianying")
DATA, REPORTS, FIGURES = ROOT / "data", ROOT / "reports", ROOT / "figures"
TOP_EACH, TOP_MERGED, SEED = 20, 50, 42


def load_inputs():
    # train.csv 只读取构建召回所需字段，避免加载 600MB 特征宽表。
    use = ["user_id", "movie_id", "rating", "timestamp", "event_id", "label", "sample_source"]
    train = pd.read_csv(DATA / "train.csv", usecols=use)
    valid = pd.read_pickle(DATA / "features_valid.pkl")
    movies = pd.read_pickle(DATA / "df_movies.pkl")
    observed = train.loc[train.sample_source.eq("observed_positive") | train.sample_source.eq("observed_low_rating")].copy()
    observed = observed.loc[observed.rating.notna()].sort_values(["user_id", "timestamp", "event_id"], kind="stable")
    print(f"[load] observed train events={len(observed):,}, valid rows={len(valid):,}", flush=True)
    return train, valid, movies, observed


def make_queries(observed, valid):
    by_user = {int(u): g for u, g in observed.groupby("user_id", sort=False)}
    queries = {}
    targets = {}
    for r in valid.itertuples(index=False):
        uid, ts = int(r.user_id), int(r.timestamp)
        g = by_user.get(uid)
        if g is None:
            items, ratings = [], []
        else:
            t = g.timestamp.to_numpy(np.int64); stop = np.searchsorted(t, ts, side="left")
            items = g.movie_id.iloc[:stop].astype(int).tolist(); ratings = g.rating.iloc[:stop].astype(float).tolist()
        queries[uid] = {"time": ts, "items": items, "ratings": ratings}
        if int(r.label) == 1: targets[uid] = int(r.movie_id)
    return queries, targets


def popular_recall(observed, queries, k=TOP_EACH):
    e = observed.sort_values("timestamp"); times=e.timestamp.to_numpy(np.int64); mids=e.movie_id.to_numpy(np.int32)
    ordered=sorted(queries.items(), key=lambda x:x[1]["time"]); counts=Counter(); pos=0; out={}
    for uid,q in ordered:
        while pos<len(times) and times[pos] < q["time"]: counts[int(mids[pos])] += 1; pos += 1
        seen=set(q["items"]); out[uid]=[m for m,_ in counts.most_common() if m not in seen][:k]
    print(f"[popular] users={len(out):,}, candidates={sum(map(len,out.values())):,}", flush=True)
    return out


def itemcf_recall(observed, queries, k=TOP_EACH, max_history=30):
    # 稀疏 Counter 共现，单用户只保留最近 max_history 个去重电影。
    co=defaultdict(Counter); freq=Counter()
    for uid,g in observed.groupby("user_id", sort=False):
        items=list(dict.fromkeys(g.movie_id.astype(int).tolist()[-max_history:]))
        for a in items: freq[a]+=1
        for i,a in enumerate(items):
            for b in items[i+1:]: co[a][b]+=1; co[b][a]+=1
    out={}
    for uid,q in queries.items():
        seen=set(q["items"]); score=Counter()
        for a in q["items"][-max_history:]:
            for b,c in co.get(a,{}).items():
                if b not in seen: score[b]+=c/math.sqrt(max(1,freq[a]*freq[b]))
        out[uid]=[m for m,_ in score.most_common(k)]
    print(f"[itemcf] sparse_pairs={sum(len(x) for x in co.values()):,}, candidates={sum(map(len,out.values())):,}", flush=True)
    return out


class Tower(nn.Module):
    def __init__(self, nu, ni, dim=32):
        super().__init__(); self.uemb=nn.Embedding(nu,16); self.iemb=nn.Embedding(ni+1,16, padding_idx=0)
        self.uf=nn.Sequential(nn.Linear(16+2+16,64),nn.ReLU(),nn.Linear(64,dim))
        self.if_=nn.Sequential(nn.Linear(16+32+2,64),nn.ReLU(),nn.Linear(64,dim))
    def user_vec(self, uid, stats, seq):
        seqv=self.iemb(seq).mean(1); return nn.functional.normalize(self.uf(torch.cat([self.uemb(uid),stats,seqv],1)),dim=1)
    def item_vec(self, iid, bge, stats):
        return nn.functional.normalize(self.if_(torch.cat([self.iemb(iid+1),bge,stats],1)),dim=1)


def tower_recall(train, valid, observed, queries, movies, k=TOP_EACH, max_samples=50000, batch_size=2048):
    torch.manual_seed(SEED); np.random.seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    users=sorted(set(observed.user_id.astype(int))|set(valid.user_id.astype(int))); items=sorted(movies.movie_id.astype(int)); um={u:i for i,u in enumerate(users)}; im={m:i for i,m in enumerate(items)}
    payload=pd.read_pickle(DATA/"movie_bge_emb.pkl"); emb={int(a):np.asarray(b,np.float32) for a,b in payload["emb_map"].items()}
    movies_i=movies.set_index("movie_id"); freq=observed.movie_id.value_counts(); item_stats=np.array([[math.log1p(freq.get(m,0))/10,len(str(movies_i.loc[m].genres).split("|"))/5] for m in items],np.float32); item_bge=np.stack([emb[m] for m in items]).astype(np.float32)
    # 为训练行和验证用户构造严格早于 timestamp 的最近10条序列；只使用训练真实行为。
    byu={int(u):g for u,g in observed.groupby("user_id",sort=False)}
    sample=train.iloc[np.linspace(0,len(train)-1,min(max_samples,len(train)),dtype=int)].copy()
    def row_inputs(row):
        g=byu.get(int(row.user_id));
        if g is None:return [0]*10,0.0,0.0
        stop=np.searchsorted(g.timestamp.to_numpy(np.int64),int(row.timestamp),side="left"); h=g.iloc[max(0,stop-10):stop]; seq=[im.get(int(x),-1)+1 for x in h.movie_id.tolist()]; return [0]*(10-len(seq))+seq,float(stop),float(h.rating.mean() if len(h) else 0)
    rows=[row_inputs(r) for r in sample.itertuples(index=False)]; seq=np.array([x[0] for x in rows],np.int64); stats=np.array([[x[1]/1000,x[2]/5] for x in rows],np.float32); uid=np.array([um[int(x)] for x in sample.user_id],np.int64); iid=np.array([im[int(x)] for x in sample.movie_id],np.int64); y=sample.label.to_numpy(np.float32)
    model=Tower(len(users),len(items)).to(device); opt=torch.optim.Adam(model.parameters(),lr=2e-3); lossfn=nn.BCEWithLogitsLoss(); order=np.arange(len(sample))
    for epoch in range(1):
        np.random.default_rng(SEED).shuffle(order); total=0.0
        for st in range(0,len(order),batch_size):
            ix=order[st:st+batch_size]; U=torch.tensor(uid[ix],device=device); I=torch.tensor(iid[ix],device=device); S=torch.tensor(stats[ix],device=device); Q=torch.tensor(seq[ix],device=device); B=torch.tensor(item_bge[iid[ix]],device=device); T=torch.tensor(item_stats[iid[ix]],device=device); Y=torch.tensor(y[ix],device=device)
            opt.zero_grad(); logits=(model.user_vec(U,S,Q)*model.item_vec(I,B,T)).sum(1)*10; loss=lossfn(logits,Y); loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
        print(f"[two_tower] device={device}, epoch={epoch+1}, samples={len(sample):,}, loss={total/len(sample):.5f}",flush=True)
    # 物品向量一次性仅 3,883 条；用户向量批量计算。
    with torch.no_grad():
        IV=model.item_vec(torch.arange(len(items),device=device),torch.tensor(item_bge,device=device),torch.tensor(item_stats,device=device)).cpu().numpy()
    out={}; valid_by={int(r.user_id):r for r in valid.itertuples(index=False)}
    with torch.no_grad():
        for st in range(0,len(users),512):
            us=users[st:st+512]; s=[]; qseq=[]
            for u in us:
                q=queries[u]; h=q["items"][-10:]; qseq.append([0]*(10-len(h))+[im.get(x,-1)+1 for x in h]); s.append([len(q["items"])/1000,(np.mean(q["ratings"][-10:]) if q["ratings"] else 0)/5])
            UV=model.user_vec(torch.tensor([um[u] for u in us],device=device),torch.tensor(s,dtype=torch.float32,device=device),torch.tensor(qseq,device=device)).cpu().numpy(); scores=UV@IV.T
            for j,u in enumerate(us):
                seen=set(queries[u]["items"]); order=np.argsort(-scores[j]); out[u]=[items[x] for x in order if items[x] not in seen][:k]
    print(f"[two_tower] users={len(out):,}, candidates={sum(map(len,out.values())):,}",flush=True); return out


def semantic_recall(queries,k=TOP_EACH):
    payload=pd.read_pickle(DATA/"movie_bge_emb.pkl"); emb={int(a):np.asarray(b,np.float32) for a,b in payload["emb_map"].items()}; ids=sorted(emb); mat=np.stack([emb[x] for x in ids]); mat/=np.linalg.norm(mat,axis=1,keepdims=True).clip(1e-8); out={}
    for u,q in queries.items():
        h=[x for x in q["items"] if x in emb]; seen=set(q["items"])
        if not h:out[u]=[];continue
        v=np.mean(np.stack([emb[x] for x in h]),0);v/=max(1e-8,np.linalg.norm(v)); order=np.argsort(-(mat@v));out[u]=[ids[i] for i in order if ids[i] not in seen][:k]
    print(f"[semantic] users={len(out):,}, candidates={sum(map(len,out.values())):,}",flush=True); return out


def evaluate(channels,targets,observed):
    freq=observed.movie_id.value_counts(); threshold=float(freq.quantile(.2)); tail=set(freq[freq<=threshold].index.astype(int)); merged={u:list(dict.fromkeys(sum((channels[n].get(u,[]) for n in channels),[])))[:TOP_MERGED] for u in channels["popular"]}; allc={**channels,"merged":merged}; tail_target_count=sum(x in tail for x in targets.values())
    report={"eval":"valid positive targets only","long_tail_definition":f"training observed count <= 20th percentile ({threshold:.2f})","tail_items":len(tail),"tail_target_count":tail_target_count,"channels":{}}
    for name,c in allc.items():
        report["channels"][name]={"candidate_count_total":sum(map(len,c.values())),"recall":{},"long_tail_candidate_coverage":sum(any(x in tail for x in c.get(u,[])) for u in targets)/len(targets),"long_tail_target_recall":sum(targets[u] in tail and targets[u] in c.get(u,[]) for u in targets)/max(1,sum(x in tail for x in targets.values()))}
        for k in (10,20,50):report["channels"][name]["recall"][f"@{k}"]=sum(targets[u] in c.get(u,[])[:k] for u in targets)/len(targets)
    return merged,report


def main():
    REPORTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True);train,valid,movies,observed=load_inputs();queries,targets=make_queries(observed,valid); print(f"[query] users={len(queries):,}, positive targets={len(targets):,}",flush=True)
    channels={"popular":popular_recall(observed,queries),"itemcf":itemcf_recall(observed,queries),"two_tower":tower_recall(train,valid,observed,queries,movies),"semantic":semantic_recall(queries)}
    merged,report=evaluate(channels,targets,observed); pd.to_pickle({"merged":merged,"channels":channels},DATA/"recall_candidates_valid_channels.pkl"); pd.to_pickle(merged,DATA/"recall_candidates_valid.pkl"); (REPORTS/"recall_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,5));ks=(10,20,50)
    for n in channels:ax.plot(ks,[report["channels"][n]["recall"][f"@{k}"] for k in ks],marker="o",label=n)
    ax.plot(ks,[report["channels"]["merged"]["recall"][f"@{k}"] for k in ks],marker="o",linewidth=3,label="merged");ax.set(xlabel="K",ylabel="Recall",title="Four-channel recall on valid");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"recall_metrics.png",dpi=160);plt.close(fig);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

# 面试说明：热门召回稳定、适合冷启动但热门偏置明显；ItemCF 利用用户共现、可解释，
# 但对新物品和稀疏用户较弱；双塔把用户画像、历史序列和 BGE 融合，线上点积快，
# 但受训练样本和负采样影响；BGE 语义召回不依赖协同行为，适合内容冷启动，
# 但 MovieLens 标题很短。四路先并行保留候选，粗排/精排/重排留到后续阶段。
