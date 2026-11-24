r"""第二阶段：显式反馈转二分类、每用户 leave-last-two-out、训练负采样。
运行：python D:\xiangmu-dianying\sample_construct.py
依赖：pandas、numpy、matplotlib。输出 data/*.csv、reports/*.json、figures/*.png。

业务备注：MovieLens 是显式反馈数据，评分低于 4 模拟负样本；
工业上线应使用曝光未点击日志。随机未交互电影也不是真实曝光负例。
本阶段是离线 CTR 代理任务，不能将评估结果解释为真实点击率或线上收益。
"""
from pathlib import Path
import argparse
import json
import math
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'D:\xiangmu-dianying')
DATA_DIR = ROOT / 'data'
REPORT_DIR = ROOT / 'reports'
FIGURE_DIR = ROOT / 'figures'


def split_observed(ratings):
    """先切分真实交互，绝不让合成负样本参与最后两条的定位。

    timestamp 并列时按 movie_id、原始 event_id 升序确定可复现顺序，
    该规则不代表真实因果顺序。后续行为序列应使用 timestamp < 当前时刻。
    每用户留出不等于全局时间切分，不能完全消除跨用户时间穿越。
    """
    required = ['user_id', 'movie_id', 'rating', 'timestamp']
    if not set(required).issubset(ratings.columns):
        raise ValueError(f'评分表缺少字段：{required}')
    df = ratings[required].copy().reset_index(drop=True)
    if df.empty or df.isna().any().any() or not df.rating.isin(range(1, 6)).all():
        raise ValueError('评分表为空、含空值或包含非法评分')
    if df.duplicated(required).any():
        raise ValueError('评分表仍含完全重复记录，请先运行第一阶段')
    df['event_id'] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(['user_id', 'timestamp', 'movie_id', 'event_id'], kind='stable')
    group = df.groupby('user_id', sort=False)
    size = group.user_id.transform('size')
    if size.min() < 3:
        raise ValueError('存在少于 3 条交互的用户，无法同时保留训练/验证/测试')
    position = group.cumcount()
    df['label'] = df.rating.ge(4).astype('int8')
    df['sample_source'] = np.where(df.label.eq(1), 'observed_positive', 'observed_low_rating')
    # 合成样本通过此字段关联正样本上下文，不能把它作为模型输入。
    df['anchor_event_id'] = df.event_id
    masks = {'train': position.lt(size - 2), 'valid': position.eq(size - 2),
             'test': position.eq(size - 1)}
    splits = {name: df.loc[mask].copy().reset_index(drop=True) for name, mask in masks.items()}
    return splits


def sample_global_negatives(train, seed):
    """训练集保留全部真实正/负例，补到负例数 = 正例数 * 3。

    候选池为全局训练日志中首次出现时间 < anchor_timestamp 的电影，
    均匀抽样而非按热门度加权。最早出现时间仅作为可用性门槛，不是特征。
    它是可观测目录的保守代理，会排除尚未有交互的真实可用电影。
    不查用户验证/测试行为，也不利用其未来训练行为排除候选电影。
    因此随机负例可能在未来被用户喜欢：这是隐式负采样的假负例限制。
    """
    positive_count = int(train.label.sum())
    low_count = int(train.label.eq(0).sum())
    target = positive_count * 3 - low_count
    if positive_count == 0 or target < 0:
        raise ValueError('无训练正例或真实负例已超过 1:3；无法在保留真实数据的同时满足目标比例')
    rng = np.random.default_rng(seed)
    # 只使用训练集定义候选池，绝不读验证/测试集生成训练样本。
    first_seen = train.groupby('movie_id').timestamp.min().sort_values(kind='stable')
    pool_ids = first_seen.index.to_numpy(dtype=np.int64)
    pool_times = first_seen.to_numpy(dtype=np.int64)
    movie_first = first_seen.to_dict()
    remaining, anchors_left = target, positive_count
    rows, skipped = [], 0
    # 逐用户、逐时间块维护历史，同一时间的真实交互统一排除，
    # 避免同一个用户在同一时刻的真实交互又被合成为负例。
    for user_id, user in train.groupby('user_id', sort=False):
        seen = set()
        # Python 迭代时间块，避免为每个时间戳创建 pandas 子表。
        for timestamp, block_iter in groupby(user.itertuples(index=False),
                                             key=lambda row: row.timestamp):
            block = list(block_iter)
            seen.update(int(row.movie_id) for row in block)
            pool_size = int(np.searchsorted(pool_times, timestamp, side='left'))
            blocked_count = sum(movie_first[mid] < timestamp for mid in seen)
            available = pool_size - blocked_count
            for anchor in block:
                if anchor.label != 1:
                    continue
                # 将已有低分负例计入总预算，不在每个正例后机械补 3 条。
                quota = min(math.ceil(remaining / anchors_left), max(0, available))
                anchors_left -= 1
                if quota == 0:
                    skipped += 1
                    continue
                chosen = set()
                attempts = 0
                while len(chosen) < quota:
                    mid = int(pool_ids[rng.integers(pool_size)])
                    if mid not in seen:
                        chosen.add(mid)
                    attempts += 1
                    # 高活跃用户可用候选很少时，退化为显式候选集，避免拒绝采样长循环。
                    if attempts > max(50, quota * 10) and len(chosen) < quota:
                        candidates = np.array([m for m in pool_ids[:pool_size]
                                               if m not in seen and m not in chosen])
                        chosen.update(int(m) for m in rng.choice(
                            candidates, size=quota - len(chosen), replace=False))
                for mid in sorted(chosen):
                    rows.append((int(user_id), mid, int(timestamp), int(anchor.event_id)))
                remaining -= quota
    if remaining:
        raise ValueError(f'历史可用候选不足，还缺 {remaining} 条负例；未放宽时序约束')
    synthetic = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'timestamp', 'anchor_event_id'])
    synthetic['event_id'] = -np.arange(1, len(synthetic) + 1, dtype=np.int64)
    # 没有真实评分，绝不能伪造 rating=0；CSV 中保存为空值。
    synthetic['rating'] = pd.Series(pd.NA, index=synthetic.index, dtype='Int8')
    synthetic['label'] = np.int8(0)
    synthetic['sample_source'] = 'random_unobserved'
    result = pd.concat([train, synthetic[train.columns]], ignore_index=True)
    result = result.sort_values(['user_id', 'timestamp', 'anchor_event_id', 'event_id'],
                                kind='stable').reset_index(drop=True)
    return result, {'seed': seed, 'added_random_negatives': len(synthetic),
                    'anchors_with_zero_quota': skipped,
                    'pool_rule': 'uniform among training movies first seen strictly before anchor'}


def validate_and_report(splits, train, sampling):
    """验证分割隔离、留出数量、比例和每条合成负例的时间约束。"""
    real_train, valid, test = (splits[k] for k in ('train', 'valid', 'test'))
    real_sets = [set(d.event_id) for d in (real_train, valid, test)]
    if any(real_sets[i] & real_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise RuntimeError('真实事件发生跨集合重叠')
    if not valid.user_id.is_unique or not test.user_id.is_unique:
        raise RuntimeError('验证/测试集未做到每用户恰好一条')
    train_end = real_train.groupby('user_id').timestamp.max()
    valid_time = valid.set_index('user_id').timestamp.reindex(train_end.index)
    test_time = test.set_index('user_id').timestamp.reindex(train_end.index)
    if not ((train_end <= valid_time) & (valid_time <= test_time)).all():
        raise RuntimeError('用户内时间顺序错误')
    observed = train.loc[train.sample_source.ne('random_unobserved')]
    if set(observed.event_id) != real_sets[0] or len(observed) != len(real_train):
        raise RuntimeError('训练真实事件丢失或重复')
    if int(train.label.eq(0).sum()) != 3 * int(train.label.eq(1).sum()):
        raise RuntimeError('训练集不满足正负 1:3')
    syn = train.loc[train.sample_source.eq('random_unobserved')]
    anchors = real_train.set_index('event_id')
    if not (syn.timestamp.to_numpy() == anchors.loc[syn.anchor_event_id, 'timestamp'].to_numpy()).all():
        raise RuntimeError('合成样本与锚点时间不一致')
    if not anchors.loc[syn.anchor_event_id, 'label'].eq(1).all():
        raise RuntimeError('合成样本锚点不是正例')
    known = real_train.groupby('movie_id').timestamp.min()
    if not (syn.movie_id.map(known) < syn.timestamp).all():
        raise RuntimeError('合成电影尚未进入历史候选池')
    user_first = real_train.groupby(['user_id', 'movie_id']).timestamp.min().rename('user_first_time')
    check = syn.join(user_first, on=['user_id', 'movie_id'])
    if (check.user_first_time.notna() & check.user_first_time.le(check.timestamp)).any():
        raise RuntimeError('负采样命中了该用户当前或过去的交互')
    if syn.duplicated(['anchor_event_id', 'movie_id']).any():
        raise RuntimeError('同一锚点重复采样电影')
    report = {'sampling': sampling, 'splits': {}, 'boundary_timestamp_ties': {
        'train_valid': int(train_end.eq(valid_time).sum()),
        'valid_test': int(valid_time.eq(test_time).sum())},
        'notes': [
            'Explicit-rating binary proxy; not real exposure CTR.',
            'Train ratio 1:3 includes observed low ratings; validation/test remain observed only.',
            'No random split. Tie order is deterministic, not causal.',
            'Per-user leave-out does not ensure global chronological isolation.',
            'Future positives are not consulted during sampling; false negatives can occur.',
            'Use timestamp < query time for histories. Fit feature transforms on training only.',
            'Sampling is uniform over historically observed training catalog, not all raw movies.',
            'Uniform sampling reduces popularity weighting, but tail improvement requires evaluation.',
        ]}
    for name, df in {'train': train, 'valid': valid, 'test': test}.items():
        report['splits'][name] = {'rows': len(df), 'positive': int(df.label.sum()),
            'negative': int(df.label.eq(0).sum()), 'positive_rate': float(df.label.mean()),
            'users': int(df.user_id.nunique()), 'movies': int(df.movie_id.nunique())}
    return report


def plot_report(report, real_train, train):
    """画切分后的标签计数和训练电影频次，后者仅为采样诊断。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    names = ['train', 'valid', 'test']
    x = np.arange(3)
    axes[0].bar(x - .18, [report['splits'][n]['positive'] for n in names], .36, label='Positive')
    axes[0].bar(x + .18, [report['splits'][n]['negative'] for n in names], .36, label='Negative')
    axes[0].set(xticks=x, xticklabels=names, yscale='log', title='Label counts (log scale)')
    axes[0].legend()
    sources = train.sample_source.value_counts()
    axes[1].bar(sources.index, sources.values, color=['#3977ae', '#519c91', '#d07932'])
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].set(title='Training sample sources', ylabel='Samples')
    counts = real_train.groupby('movie_id').size().sort_values(ascending=False)
    sampled = train.loc[train.sample_source.eq('random_unobserved')].groupby('movie_id').size()
    axes[2].plot(range(1, len(counts)+1), counts, label='Observed train events')
    axes[2].plot(range(1, len(counts)+1), sampled.reindex(counts.index, fill_value=0),
                 alpha=.7, label='Random negatives')
    axes[2].set(title='Sampling diagnosis', xlabel='Movie rank by train frequency', ylabel='Samples')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'sample_construct_quality.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def main(seed=42):
    for folder in (DATA_DIR, REPORT_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_pickle(DATA_DIR / 'df_ratings.pkl')
    splits = split_observed(ratings)
    train, sampling = sample_global_negatives(splits['train'], seed)
    report = validate_and_report(splits, train, sampling)
    # 此阶段输出事件字段和审计字段；用户/电影特征由下一阶段按 ID 关联。
    # rating、label、event_id、anchor_event_id、sample_source 不作为模型输入特征。
    for name, df in {'train': train, 'valid': splits['valid'], 'test': splits['test']}.items():
        df.to_csv(DATA_DIR / f'{name}.csv', index=False, encoding='utf-8-sig')
    plot_report(report, splits['train'], train)
    with (REPORT_DIR / 'sample_construct_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'CSV saved to {DATA_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42, help='仅控制负采样，不随机切分数据')
    main(seed=parser.parse_args().seed)
