"""有意义的时序回归检查：任意查询顺序、跨用户未来记录、同时间戳、合成负例。"""
import sys
sys.path.insert(0, r'D:\xiangmu-dianying')
import features as f
import pandas as pd
import numpy as np

genres={1:('A',),2:('A','B'),3:('B',),4:('A',)}
events=pd.DataFrame([(1,1,5,100,1,'observed_positive'),
                     (2,1,2,10,2,'observed_low_rating'),
                     (2,2,4,20,3,'observed_positive'),
                     (2,3,1,20,4,'observed_low_rating'),
                     (2,4,np.nan,15,-1,'random_unobserved')],
                    columns=['user_id','movie_id','rating','timestamp','event_id','sample_source'])
queries=pd.DataFrame([(2,1,11),(2,2,20),(1,1,101),(2,2,21),(2,1,5)],
                     columns=['user_id','movie_id','timestamp'])
actual=f.historical_features(queries,events,genres)
real=events[events.sample_source.ne('random_unobserved')]
for i,q in enumerate(queries.itertuples(index=False)):
    past=real[real.timestamp.lt(q.timestamp)]
    own=past[past.user_id.eq(q.user_id)].sort_values(['timestamp','movie_id','event_id'])
    assert actual['movie_popularity'][i]==past.movie_id.eq(q.movie_id).sum()
    assert actual['user_activity'][i]==len(own)
    assert actual['history_movie_ids'][i]==own.movie_id.tail(20).tolist()
    ratings=[]
    for r in own.itertuples():
        ratings.extend([r.rating]*len(set(genres[r.movie_id])&set(genres[q.movie_id])))
    assert np.isclose(actual['user_genre_mean_rating'][i],np.mean(ratings) if ratings else 0)
assert actual['movie_popularity'][0]==1  # user 1 在 t=100 的未来事件不能污染 t=11
assert actual['user_activity'][1]==1  # 同时刻 t=20 的两个真实事件都排除
print('PASS: all historical features match brute-force strict-time reference.')
