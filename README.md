# MovieLens-1M 多路召回 + DIN 精排 + MMR 重排

面向搜推算法实习简历的轻量推荐系统项目。项目完整复现了离线推荐链路：

```text
数据清洗 → 时序样本构建 → 特征工程 → 多路召回 → DIN 精排 → MMR 重排
```

项目重点是推荐系统模块分工、严格时序隔离、长尾诊断和可解释的消融实验。模型规模控制在普通笔记本 CPU 可运行范围，DIN 支持 CUDA；召回中的双塔会抽样训练，MMR 使用 CPU 候选池内计算。

## 项目结果

### 多路召回（valid）

每路召回 Top20，四路合并去重后每用户最多 50 个候选。

| 通道 | Recall@10 | Recall@20 | 长尾候选覆盖率 |
|---|---:|---:|---:|
| 时序热门 | 0.05557 | 0.09478 | 0.00000 |
| ItemCF | 0.11142 | 0.18505 | 0.02567 |
| 双塔 | 0.01580 | 0.03216 | 0.04880 |
| BGE 语义 | 0.01862 | 0.03272 | 0.58872 |
| 多路合并 | 0.05557 | 0.09478 | 0.08829 |

ItemCF 是当前场景的主力个性化通道；BGE 语义通道整体召回较低，但长尾候选覆盖明显领先。长尾物品如果没有进入召回候选池，后续精排无法恢复。

### DIN 精排消融（valid 候选）

| 实验 | AUC | GAUC | LogLoss |
|---|---:|---:|---:|
| DIN_base | 0.55742 | 0.54158 | 0.65029 |
| DIN_BGE | 0.55217 | 0.53851 | 0.64217 |

加入 BGE 后 LogLoss 降低，概率校准有所改善；AUC、GAUC 小幅下降。可能原因是 MovieLens 标题较短、语义信息有限，BGE 与协同行为特征存在冗余，且本项目采用了轻量训练设置。valid 长尾候选子集没有足够的正负混合样本，长尾 AUC/GAUC 报告为 `null`。

### MMR 重排消融（valid）

| λ | Recall@10 | Recall@20 | 平均两两余弦相似度 |
|---:|---:|---:|---:|
| 0.4 | 0.05642 | 0.11001 | 0.06515 |
| 0.6 | 0.05162 | 0.10691 | 0.10173 |
| 0.8 | 0.05219 | 0.10042 | 0.13008 |
| 1.0 | 0.05614 | 0.09591 | 0.14213 |

λ 越小，多样性越强。当前候选池中 λ=0.4 的 Recall@20 反而最高，说明高分候选里存在语义相似的负样本扎堆；这个现象依赖当前候选分布，不代表 MMR 在所有数据上都必然提升相关性。

## 数据集与时序协议

使用 MovieLens-1M：

- 用户 6,040；电影 3,883；评分事件 1,000,209；
- 原始文件位于 `D:\xiangmu-dianying\ml-1m\`：`ratings.dat`、`movies.dat`、`users.dat`；
- 评分 `rating >= 4` 转为正标签，低分作为显式反馈场景下的负样本代理；工业 CTR 应使用曝光未点击日志；
- 每个用户按 `timestamp` 排序，最后一条测试、倒数第二条验证，其余训练；不随机切分；
- 统计特征和行为序列严格使用 `history_timestamp < sample_timestamp`；
- test 文件已生成但保留给最终全链路验收，本项目报告只使用 valid；
- 长尾定义为训练真实交互次数不超过电影交互数第 20 百分位，阈值 23 次，共 759 部电影。

时间戳并列时使用稳定的 `movie_id/event_id` 顺序保证可复现；并列不代表真实因果先后。每用户留出协议也不等价于全局时间切分。

## 环境依赖

```powershell
pip install pandas numpy torch scikit-learn matplotlib modelscope FlagEmbedding
```

本项目已在 Python 3.12、pandas 2.2、PyTorch 2.9、scikit-learn 1.5 环境验证。CUDA 可用时，`din_rank.py` 自动使用 GPU；当前双塔和 MMR 不要求 GPU。

## 目录结构

```text
D:\xiangmu-dianying\
├── ml-1m\                         # 原始 dat 文件
├── data1_preparation.py            # 第一步：清洗与宽表
├── sample_construct.py             # 第二步：标签、时序切分、负采样
├── features.py                     # 第三步：统计、序列、语义特征
├── download_bge.py                 # ModelScope 离线模型下载
├── gen_bge_emb.py                  # BGE 768维 + PCA 32维
├── recall.py                       # 第四步：四路召回
├── din_rank.py                     # 第五步：DIN 精排消融
├── rerank_mmr.py                   # 第六步：MMR 重排
├── bge-base-en-v1.5\              # 本地离线 BGE 模型
├── data\                           # 表、特征、候选和向量
├── reports\                        # JSON 质量与实验报告
├── figures\                        # matplotlib 图表
└── work\                           # 临时缓存与验证脚本
```

核心输出：

```text
data/df_ratings.pkl, df_movies.pkl, df_users.pkl, df_wide.pkl
data/train.csv, valid.csv, test.csv
data/features_train.pkl, features_valid.pkl, features_test.pkl
data/movie_bge_emb.pkl
data/recall_candidates_valid.pkl
data/din_score_valid.pkl
data/mmr_rerank_valid.pkl
reports/recall_report.json, din_report.json, mmr_report.json
figures/recall_metrics.png, din_auc_curve.png, mmr_tradeoff.png
```

## 复现步骤

按阶段依次运行：

```powershell
python D:\xiangmu-dianying\data1_preparation.py
python D:\xiangmu-dianying\sample_construct.py
```

BGE 语义特征只需首次准备模型时运行：

```powershell
python D:\xiangmu-dianying\download_bge.py
python D:\xiangmu-dianying\gen_bge_emb.py
python D:\xiangmu-dianying\features.py
```

`download_bge.py` 从 ModelScope 的 `AI-ModelScope/bge-base-en-v1.5` 下载到 `D:\xiangmu-dianying\bge-base-en-v1.5`。之后 `gen_bge_emb.py` 设置离线环境变量，只读取本地模型，不再访问 Hugging Face。

然后运行主链路：

```powershell
python D:\xiangmu-dianying\recall.py
python D:\xiangmu-dianying\din_rank.py
python D:\xiangmu-dianying\rerank_mmr.py
```

后面三个脚本只消费已有中间产物：

- `recall.py` 只评估 valid，不读取 test；
- `din_rank.py` 读取召回候选做 DIN_base / DIN_BGE 精排；
- `rerank_mmr.py` 读取 DIN 分数和本地 32 维向量做 MMR。

不要为了复现后续阶段重新运行前面的数据处理，除非原始数据或中间产物确实发生变化。

## 模块面试讲解

### 多路召回

热门召回稳定、可作为冷启动兜底，但热门偏置明显；ItemCF 利用用户共现，解释性强，是本项目主要个性化通道；双塔适合在线向量点积和多特征融合，但受训练样本、负采样和轻量训练影响；BGE 语义召回不依赖协同行为，适合内容冷启动和长尾扩展，但 MovieLens 标题信息有限。

### DIN 精排

DIN 用目标电影和用户最近 20 条行为做 Target Attention，动态选择与当前候选相关的兴趣。基线不使用 BGE，实验组加入 BGE PCA 32 维向量，通过 AUC、LogLoss、GAUC 和长尾子集对比语义信号价值。精排不能弥补召回未覆盖的物品。

### MMR 重排

MMR 在 DIN 相关性分数上加入已选电影相似度惩罚。首页、内容流、探索场景通常适合用较小 λ 换取更丰富的内容覆盖；强意图搜索或专题页应提高 λ。真实业务还应结合曝光、点击、停留和转化等线上指标调参。

## 关键工程限制

1. MovieLens 是显式评分数据，不是曝光点击日志，CTR 和 LogLoss 只能作为离线代理。
2. 用户画像没有历史更新时间，实验将其视为静态画像；这是离线数据的可用性假设。
3. 用户内最后两条留出不是全局时间切分，跨用户时间交错仍可能存在。
4. 随机未交互电影可能是未来正反馈，属于隐式负采样中的假负例风险。
5. BGE 只做语义消融，不能把有限标题文本解释成完整内容理解。

