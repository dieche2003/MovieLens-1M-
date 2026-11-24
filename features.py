r"""第三阶段：无泄漏特征工程。

运行：python D:\xiangmu-dianying\features.py
先运行 gen_bge_emb.py 生成本地向量；此脚本只读 data/movie_bge_emb.pkl，完全离线。
--skip-semantic 仅供无语义基线使用，显式生成零占位列。输出在 data/、reports/、figures/。

MovieLens 是显式反馈数据；此处低分已经由上阶段转成负标签。BGE 仅作为
语义消融对照，MovieLens 标题信息有限，不应把语义向量当作主实验特征。
"""
from pathlib import Path
import argparse, json
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\xiangmu-dianying")
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def load_inputs():
    movies = pd.read_pickle(DATA / "df_movies.pkl")
    users = pd.read_pickle(DATA / "df_users.pkl")
    splits = {n: pd.read_csv(DATA / f"{n}.csv") for n in ("train", "valid", "test")}
    for n, d in splits.items():
        d["user_id"] = d.user_id.astype(np.int32); d["movie_id"] = d.movie_id.astype(np.int32)
        d["timestamp"] = d.timestamp.astype(np.int64); d["label"] = d.label.astype(np.int8)
    return movies, users, splits


def genre_tokens(value):
    return tuple(str(value).split("|"))


def historical_features(df, allowed_events, movie_genres, max_history=20):
    """用 searchsorted(side='left') 保证严格小于目标时刻，且不依赖查询行顺序。

    全局电影热度独立按电影时间轴查询，不能在逐用户遍历时累加全局计数。
    用户活跃度/序列包含所有真实评分（含低分），不包含合成负样本。
    多 genre 平均按历史 genre 归属加权：一个历史电影匹配两个目标 genre 时计两次。
    未知历史均值设 0；它是缺历史标记，不是从全量评分估计的先验。
    """
    events = allowed_events.loc[allowed_events.sample_source.ne("random_unobserved")].copy()
    events = events.sort_values(["user_id", "timestamp", "movie_id", "event_id"], kind="stable")
    size = len(df)
    popularity = np.zeros(size, dtype=np.int32)
    activity = np.zeros(size, dtype=np.int32)
    mean_rating = np.zeros(size, dtype=np.float32)
    histories = [[] for _ in range(size)]
    query_times = df.timestamp.to_numpy(dtype=np.int64)
    query_movies = df.movie_id.to_numpy(dtype=np.int64)
    movie_times = {int(mid): np.sort(g.timestamp.to_numpy(dtype=np.int64))
                   for mid, g in events.groupby("movie_id", sort=False)}
    for mid, positions in df.groupby("movie_id", sort=False).indices.items():
        times = movie_times.get(int(mid), np.array([], dtype=np.int64))
        popularity[positions] = np.searchsorted(times, query_times[positions], side="left")
    user_events = {int(uid): g for uid, g in events.groupby("user_id", sort=False)}
    for uid, positions in df.groupby("user_id", sort=False).indices.items():
        past = user_events.get(int(uid))
        if past is None:
            continue
        times = past.timestamp.to_numpy(dtype=np.int64)
        mids = past.movie_id.to_numpy(dtype=np.int64)
        ratings = past.rating.to_numpy(dtype=np.float64)
        qt = query_times[positions]
        stops = np.searchsorted(times, qt, side="left")
        activity[positions] = stops
        for pos, stop in zip(positions, stops):
            histories[pos] = mids[max(0, stop - max_history):stop].tolist()
        # 对每个 genre 建立历史时间/评分前缀和，再为目标 movie 的所有 genre 查询。
        genre_indices = defaultdict(list)
        for i, mid in enumerate(mids):
            for genre in movie_genres.get(int(mid), ()):
                genre_indices[genre].append(i)
        sums = np.zeros(len(positions), dtype=np.float64)
        counts = np.zeros(len(positions), dtype=np.int64)
        for genre, indices in genre_indices.items():
            indices = np.asarray(indices, dtype=np.int64)
            ends = np.searchsorted(times[indices], qt, side="left")
            prefix = np.concatenate(([0.0], np.cumsum(ratings[indices])))
            matches = np.fromiter((genre in movie_genres.get(int(mid), ())
                                   for mid in query_movies[positions]), dtype=bool)
            sums[matches] += prefix[ends[matches]]
            counts[matches] += ends[matches]
        mean_rating[positions] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return {"movie_popularity": popularity, "user_activity": activity,
            "user_genre_mean_rating": mean_rating, "history_movie_ids": histories}


def add_bge_features(movies, skip=False, embedding_file=None):
    """只读本地向量文件，永远不实例化在线模型或联网下载。"""
    if skip:
        return {}, {"enabled": False, "reason": "--skip-semantic"}
    path = Path(embedding_file) if embedding_file else DATA / "movie_bge_emb.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}，先离线运行 gen_bge_emb.py")
    payload = pd.read_pickle(path)
    metadata = payload["metadata"]
    if not metadata.get("enabled") or metadata.get("raw_dim") != 768 or metadata.get("pca_dim") != 32:
        raise ValueError("本地向量元数据不符合 BGE 768 -> PCA 32 要求")
    mapping = payload["emb_map"]
    for mid in movies.movie_id:
        value = np.asarray(mapping.get(int(mid)), dtype=np.float32)
        if value.shape != (32,) or not np.isfinite(value).all() or not np.any(value):
            raise ValueError(f"电影 {mid} 的本地语义向量缺失或无效")
    return mapping, {**metadata, "embedding_file": str(path)}


def build_features(movies, users, splits, semantic):
    movie_genres = {int(r.movie_id): genre_tokens(r.genres) for r in movies.itertuples()}
    user_cols = ["user_id", "gender", "age", "occupation"]
    movie_cols = ["movie_id", "title", "genres"]
    user_meta = users[user_cols].copy(); user_meta["age"] = user_meta.age.astype(str); user_meta["occupation"] = user_meta.occupation.astype(str)
    movie_meta = movies[movie_cols].copy()
    # 训练样本只可看 train 真实事件；valid 可看 train；test 可看 train+valid。
    observed = {n: splits[n].loc[splits[n].sample_source != "random_unobserved"].copy() for n in splits}
    allowed = {"train": observed["train"], "valid": observed["train"],
               "test": pd.concat([observed["train"], observed["valid"]], ignore_index=True)}
    result = {}
    for name in ("train", "valid", "test"):
        d = splits[name].copy().reset_index(drop=True)
        f = historical_features(d, allowed[name], movie_genres)
        d = d.merge(user_meta, on="user_id", how="left", validate="many_to_one")
        d = d.merge(movie_meta, on="movie_id", how="left", validate="many_to_one")
        for k, v in f.items(): d[k] = v
        d["genre_count"] = d.genres.astype(str).str.count(r"\|") + 1
        if semantic:
            embedding_table = pd.DataFrame.from_dict(semantic, orient="index")
            matrix = embedding_table.reindex(d.movie_id.to_numpy()).to_numpy(dtype=np.float32)
            if matrix.shape != (len(d), 32) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} 存在缺失的电影向量")
        else:
            matrix = np.zeros((len(d), 32), dtype=np.float32)
        for j in range(32):
            d[f"bge_pca_{j:02d}"] = matrix[:, j]
        result[name] = d
    return result


def main(skip_semantic=False, embedding_file=None):
    for p in (DATA, REPORTS, FIGURES): p.mkdir(parents=True, exist_ok=True)
    movies, users, splits = load_inputs()
    semantic, sem_report = add_bge_features(movies, skip_semantic, embedding_file)
    features = build_features(movies, users, splits, semantic)
    feature_dict = {"categorical": ["user_id", "movie_id", "gender", "age", "occupation", "genres"],
                    "numeric": ["movie_popularity", "user_activity", "user_genre_mean_rating", "genre_count"],
                    "sequence": {"name": "history_movie_ids", "max_length": 20, "padding": 0},
                    "semantic": sem_report, "leakage_policy": "strict timestamp < sample timestamp; split-aware histories"}
    feature_dict["semantic_columns"] = [f"bge_pca_{j:02d}" for j in range(32)]
    feature_dict["columns"] = features["train"].columns.tolist()
    pd.to_pickle(feature_dict, DATA / "feature_dict.pkl")
    for n, d in features.items(): d.to_pickle(DATA / f"features_{n}.pkl"); d.to_csv(DATA / f"features_{n}.csv", index=False, encoding="utf-8-sig")
    report = {"semantic": sem_report, "rows": {n: len(d) for n,d in features.items()}, "notes": ["BGE is ablation only; title information is limited.", "Statistics and sequences use only split-allowed events strictly before timestamp."]}
    (REPORTS / "features_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7,4)); ax.hist(features["train"].movie_popularity, bins=50, log=True); ax.set(title="Leakage-safe historical movie popularity", xlabel="count before sample", ylabel="samples (log)"); fig.tight_layout(); fig.savefig(FIGURES / "features_quality.png", dpi=160); plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--skip-semantic", action="store_true"); ap.add_argument("--embedding-file", default=str(DATA / "movie_bge_emb.pkl")); a = ap.parse_args(); main(a.skip_semantic, a.embedding_file)
