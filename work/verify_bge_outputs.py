"""端到端产物检查：真实 BGE 覆盖、PCA 训练范围、逐样本历史暴力对照。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(r'D:\xiangmu-dianying')
DATA = ROOT / 'data'
payload = pd.read_pickle(DATA / 'movie_bge_emb.pkl')
raw = np.load(DATA / 'movie_bge_emb_768.npz')
movies = pd.read_pickle(DATA / 'df_movies.pkl')
ids = raw['movie_ids']; embeddings = raw['embeddings']
assert embeddings.shape == (len(movies), 768)
assert set(ids) == set(movies.movie_id) == set(payload['emb_map'])
assert np.isfinite(embeddings).all()
reduced = np.stack([payload['emb_map'][int(mid)] for mid in ids])
assert reduced.shape == (len(movies), 32)
assert (np.linalg.norm(reduced, axis=1) > 0).all()
assert np.allclose(payload['pca'].transform(embeddings), reduced, atol=1e-5)
train_ids = set(pd.read_csv(DATA / 'train.csv', usecols=['movie_id']).movie_id)
assert set(payload['pca_fit_movie_ids']) == train_ids & set(ids)
observed = {}
for name in ['train', 'valid', 'test']:
    source = pd.read_csv(DATA / f'{name}.csv')
    observed[name] = source[source.sample_source.ne('random_unobserved')].copy()
    del source
genres = dict(zip(movies.movie_id, movies.genres.str.split('|', regex=False)))
semantic_cols = [f'bge_pca_{i:02d}' for i in range(32)]
results = {}
for name in ['train', 'valid', 'test']:
    d = pd.read_pickle(DATA / f'features_{name}.pkl')
    allowed = observed['train'] if name != 'test' else pd.concat([observed['train'], observed['valid']])
    # 分块比较整张宽表向量，避免同时多建一个巨大的矩阵。
    for start in range(0, len(d), 100000):
        chunk = d.iloc[start:start+100000]
        expected = np.stack([payload['emb_map'][int(mid)] for mid in chunk.movie_id])
        assert np.array_equal(chunk[semantic_cols].to_numpy(dtype=np.float32), expected)
    selected = pd.concat([d.head(3), d.tail(3), d.sample(min(40,len(d)), random_state=42)])
    max_history = 0
    for q in selected.itertuples():
        past = allowed[allowed.timestamp.lt(q.timestamp)]
        own = past[past.user_id.eq(q.user_id)].sort_values(['timestamp','movie_id','event_id'])
        assert q.movie_popularity == past.movie_id.eq(q.movie_id).sum()
        assert q.user_activity == len(own)
        assert q.history_movie_ids == own.movie_id.tail(20).tolist()
        numerator, denominator = 0.0, 0
        for h in own.itertuples():
            matches = len(set(genres[h.movie_id]) & set(genres[q.movie_id]))
            numerator += h.rating * matches; denominator += matches
        expected = numerator/denominator if denominator else 0
        assert np.isclose(q.user_genre_mean_rating, expected, atol=1e-6)
        max_history = max(max_history,len(q.history_movie_ids))
    assert d.history_movie_ids.map(len).max() <= 20
    # CSV 分块复读，确保不是只更新了 pkl。
    count = 0
    for chunk in pd.read_csv(DATA/f'features_{name}.csv', usecols=['movie_id']+semantic_cols, chunksize=100000):
        expected=np.stack([payload['emb_map'][int(mid)] for mid in chunk.movie_id])
        assert np.allclose(chunk[semantic_cols], expected, atol=1e-6)
        count += len(chunk)
    assert count == len(d)
    results[name]={'rows':len(d),'semantic_rows_checked':len(d),'history_oracle_queries':len(selected),
                   'history_and_statistics':'passed','csv_reload':'passed'}
    print(name, results[name], flush=True)
    del d
assert pd.read_pickle(DATA/'feature_dict.pkl')['semantic']['enabled'] is True
results['embedding']={'raw_shape':list(embeddings.shape),'pca_shape':list(reduced.shape),
                      'pca_fit_movies':len(payload['pca_fit_movie_ids']),'zero_vectors':0}
(ROOT/'reports'/'bge_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('PASS: BGE, PCA, all semantic rows, sampled strict histories, and CSV reload.',flush=True)
