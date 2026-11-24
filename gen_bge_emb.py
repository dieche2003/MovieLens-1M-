r"""离线生成电影标题 BGE 向量和训练期拟合的 PCA。

先运行 download_bge.py 一次，然后运行 python gen_bge_emb.py。
MovieLens 标题信息有限，语义仅作为消融对照，不代表真实剧情语义。
所有推理固定 CPU float32；标题是文档，不添加 query instruction 或 passage 前缀。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\xiangmu-dianying")
DATA = ROOT / "data"
MODEL = ROOT / "bge-base-en-v1.5"


def generate(batch_size=32, threads=4):
    from FlagEmbedding import FlagModel
    for filename in ("config.json", "tokenizer.json"):
        if not (MODEL / filename).is_file():
            raise FileNotFoundError(f"离线模型缺少 {filename}，先运行 download_bge.py")
    weight = next((MODEL / f for f in ("model.safetensors", "pytorch_model.bin")
                   if (MODEL / f).is_file()), None)
    if weight is None:
        raise FileNotFoundError("模型权重尚未下载完成")
    torch.set_num_threads(threads)
    # 显式本地路径 + offline 环境变量；FlagModel 的 cls + L2 normalize 与模型配置一致。
    model = FlagModel(str(MODEL), devices="cpu", use_fp16=False,
                      normalize_embeddings=True, pooling_method="cls",
                      local_files_only=True, trust_remote_code=False)
    movies = pd.read_pickle(DATA / "df_movies.pkl").sort_values("movie_id").reset_index(drop=True)
    if not movies.movie_id.is_unique or movies.title.isna().any():
        raise ValueError("电影表主键不唯一或标题为空")
    train_ids = set()
    # PCA 仅拟合训练样本中出现的电影；分块读入避免加载 200 万行特征宽表。
    for chunk in pd.read_csv(DATA / "train.csv", usecols=["movie_id"], chunksize=200000):
        train_ids.update(int(x) for x in chunk.movie_id.unique())
    ids = movies.movie_id.to_numpy(dtype=np.int32)
    fit_mask = np.isin(ids, list(train_ids))
    print(f"Encoding {len(movies)} titles on CPU; PCA fit movies={fit_mask.sum()}", flush=True)
    raw = np.asarray(model.encode(movies.title.astype(str).tolist(),
                                 batch_size=batch_size, max_length=128), dtype=np.float32)
    if raw.shape != (len(movies), 768) or not np.isfinite(raw).all():
        raise ValueError(f"BGE 输出异常：{raw.shape}")
    if not np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-4):
        raise ValueError("BGE 向量未正确归一化")
    pca = PCA(n_components=32, svd_solver="randomized", random_state=42)
    with threadpool_limits(limits=threads):
        pca.fit(raw[fit_mask])
        reduced = pca.transform(raw).astype(np.float32)
    if not np.isfinite(reduced).all() or np.any(np.linalg.norm(reduced, axis=1) == 0):
        raise ValueError("PCA 输出非有限值或全零电影向量")
    # 模型 hash 记录本地权重来源，便于复现与审计。
    digest = hashlib.sha256()
    with weight.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata = {"enabled": True, "source": "ModelScope AI-ModelScope/bge-base-en-v1.5",
                "local_model": str(MODEL), "raw_dim": 768, "pca_dim": 32,
                "movies": len(movies), "pca_fit_movies": int(fit_mask.sum()),
                "pca_fit_scope": "unique movie IDs present in train.csv (catalog snapshot assumption)",
                "pooling": "CLS + L2 normalization", "instruction": None,
                "weight_file": weight.name, "weight_sha256": digest.hexdigest(),
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "note": "MovieLens titles are limited; BGE is ablation only."}
    payload = {"emb_map": {int(mid): reduced[i] for i, mid in enumerate(ids)},
               "pca": pca, "pca_fit_movie_ids": ids[fit_mask], "metadata": metadata}
    DATA.mkdir(exist_ok=True)
    with (DATA / "movie_bge_emb.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(DATA / "movie_bge_emb_768.npz", movie_ids=ids, embeddings=raw)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "bge_embedding_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(1, 33), np.cumsum(pca.explained_variance_ratio_))
    axes[0].set(xlabel="PCA components", ylabel="Cumulative explained variance")
    axes[1].hist(np.linalg.norm(reduced, axis=1), bins=40)
    axes[1].set(xlabel="32D embedding norm", ylabel="Movies")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "bge_embedding_quality.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    generate(args.batch_size, args.threads)
