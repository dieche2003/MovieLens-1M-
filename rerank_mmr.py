r"""第六步：DIN 候选池内 MMR 多样性重排。
运行：python D:\xiangmu-dianying\rerank_mmr.py
只读取已有 DIN 分数、valid 标签和离线 BGE PCA 32 维向量，不训练模型、不读取 test。
"""
from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(r"D:\xiangmu-dianying"); DATA=ROOT/"data"; REPORTS=ROOT/"reports"; FIGURES=ROOT/"figures"
LAMBDAS=[0.4,0.6,0.8,1.0]; TOPK=20

def load_inputs():
    scores=pd.read_pickle(DATA/"din_score_valid.pkl")
    valid=pd.read_pickle(DATA/"features_valid.pkl")
    payload=pd.read_pickle(DATA/"movie_bge_emb.pkl")
    emb={int(k):np.asarray(v,dtype=np.float32) for k,v in payload["emb_map"].items()}
    targets={int(r.user_id):int(r.movie_id) for r in valid.loc[valid.label.eq(1)].itertuples()}
    print(f"[load] users={len(scores):,}, scored candidates={sum(len(v) for v in scores.values()):,}, BGE items={len(emb):,}",flush=True)
    return scores,targets,emb

def cosine_matrix(items,emb):
    x=np.stack([emb[i] for i in items]).astype(np.float32); x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8); return x@x.T

def mmr_one(scored,emb,lam):
    # DIN 分数按用户 min-max 归一化，避免不同用户分数尺度影响 λ。
    items=[int(x[0]) for x in scored if int(x[0]) in emb]; raw=np.asarray([float(x[1]) for x in scored if int(x[0]) in emb],np.float32)
    if not items:return []
    if raw.max()>raw.min(): rel=(raw-raw.min())/(raw.max()-raw.min())
    else:rel=np.ones(len(raw),np.float32)
    sim=cosine_matrix(items,emb); selected=[]; remain=list(range(len(items)))
    while remain and len(selected)<TOPK:
        if not selected: best=max(remain,key=lambda i:(float(rel[i]),-items[i]))
        else: best=max(remain,key=lambda i:float(lam*rel[i]-(1-lam)*max(sim[i,j] for j in selected)))
        selected.append(best);remain.remove(best)
    return [items[i] for i in selected]

def diversity(items,emb):
    if len(items)<2:return 0.0
    s=cosine_matrix(items,emb); tri=s[np.triu_indices(len(items),1)]; return float(tri.mean())

def main():
    REPORTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True);scores,targets,emb=load_inputs(); results={}; output={}
    # DIN 原始列表已经按分数排序，作为 λ=1 的相关性基线。
    for lam in LAMBDAS:
        current={}; hits10=hits20=0; divs=[]
        for uid,scored in scores.items():
            ranked=mmr_one(scored,emb,lam);current[int(uid)]=ranked; target=targets.get(int(uid))
            if target is not None:
                hits10+=int(target in ranked[:10]);hits20+=int(target in ranked[:20])
            divs.append(diversity(ranked,emb))
        output[lam]=current; results[str(lam)]={"lambda":lam,"Recall@10":hits10/max(1,len(targets)),"Recall@20":hits20/max(1,len(targets)),"avg_pairwise_cosine_similarity":float(np.mean(divs)),"users":len(current),"topk":TOPK}
        print(f"[lambda={lam:.1f}] Recall@10={results[str(lam)]['Recall@10']:.5f}, Recall@20={results[str(lam)]['Recall@20']:.5f}, avg_similarity={results[str(lam)]['avg_pairwise_cosine_similarity']:.5f}",flush=True)
    pd.to_pickle(output,DATA/"mmr_rerank_valid.pkl"); report={"eval":"valid positive targets only","similarity":"cosine over offline movie_bge_emb.pkl 32D vectors","lambdas":LAMBDAS,"results":results,"notes":["lambda=1 is DIN score ordering; lower lambda trades relevance for diversity.","No model training, BGE reload or test access."]};(REPORTS/"mmr_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(7,5));
    for lam in LAMBDAS:
        r=results[str(lam)];ax.scatter(r["avg_pairwise_cosine_similarity"],r["Recall@20"],s=80,label=f"λ={lam:.1f}");ax.annotate(f"λ={lam:.1f}",(r["avg_pairwise_cosine_similarity"],r["Recall@20"]),xytext=(5,5),textcoords="offset points")
    ax.set(xlabel="Average pairwise cosine similarity (lower = more diverse)",ylabel="Recall@20",title="MMR relevance-diversity trade-off");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"mmr_tradeoff.png",dpi=160);plt.close(fig)

if __name__=="__main__":main()

# 面试说明：MMR 适用于首页推荐、内容流和需要同时展示多个兴趣方向的场景。
# λ 越大越信任 DIN 相关性，Recall 通常更稳但容易出现同质化；λ 越小越惩罚
# 与已选电影相似的候选，能扩大 genre/语义覆盖，但可能牺牲部分相关性。首页曝光、
# 长列表和用户探索场景通常值得牺牲少量相关性换多样性；强意图搜索、单一主题
# 场景则应提高 λ。实际业务还应结合覆盖率、停留时长和转化等线上指标调参。
