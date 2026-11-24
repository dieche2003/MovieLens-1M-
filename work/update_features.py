from pathlib import Path
p = Path(r'D:\xiangmu-dianying\features.py')
s = p.read_text(encoding='utf-8-sig')
start = s.index('def historical_features(')
end = s.index('\ndef build_features(', start)
replacement = '''def historical_features(df, allowed_events, movie_genres, max_history=20):
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

'''
s = s[:start] + replacement + s[end:]
old = '        for j in range(32): d[f"bge_pca_{j:02d}"] = d.movie_id.map(lambda x: semantic.get(int(x), np.zeros(32))[j]).astype("float32")'
new = '''        if semantic:
            embedding_table = pd.DataFrame.from_dict(semantic, orient="index")
            matrix = embedding_table.reindex(d.movie_id.to_numpy()).to_numpy(dtype=np.float32)
            if matrix.shape != (len(d), 32) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} 存在缺失的电影向量")
        else:
            matrix = np.zeros((len(d), 32), dtype=np.float32)
        for j in range(32):
            d[f"bge_pca_{j:02d}"] = matrix[:, j]'''
assert old in s
s = s.replace(old,new)
s = s.replace('def main(skip_semantic=False, model_name="BAAI/bge-base-en-v1.5"):', 'def main(skip_semantic=False, embedding_file=None):')
s = s.replace('add_bge_features(movies, skip_semantic, model_name)', 'add_bge_features(movies, skip_semantic, embedding_file)')
s = s.replace('    feature_dict["columns"]', '    feature_dict["semantic_columns"] = [f"bge_pca_{j:02d}" for j in range(32)]\n    feature_dict["columns"]')
s = s.replace('ap.add_argument("--bge-model", default="BAAI/bge-base-en-v1.5"); a = ap.parse_args(); main(a.skip_semantic, a.bge_model)', 'ap.add_argument("--embedding-file", default=str(DATA / "movie_bge_emb.pkl")); a = ap.parse_args(); main(a.skip_semantic, a.embedding_file)')
s = s.replace('若本机没有 BGE 权重，可先运行 --skip-semantic 生成其余特征；正式消融实验\n应提供 BAAI/bge-base-en-v1.5（768 维）并重新运行。', '先运行 gen_bge_emb.py 生成本地向量；此脚本只读 data/movie_bge_emb.pkl，完全离线。\n--skip-semantic 仅供无语义基线使用，显式生成零占位列。')
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')
print('features.py updated: strict histories, local embeddings only')
