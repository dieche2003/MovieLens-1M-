r"""MovieLens-1M：第一阶段，读取、基础清洗、宽表合并和质量检查。

运行：python D:\xiangmu-dianying\data1_preparation.py
可选导出 CSV：在命令后加 --export-csv（pickle 始终生成）。
依赖：pandas、matplotlib；此阶段无需 PyTorch。
输出目录：data/ 保存表，reports/ 保存质量报告，figures/ 保存图表。
"""
from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无图形界面的 CPU 环境也可以保存图片。
import matplotlib.pyplot as plt

ROOT = Path(r"D:\xiangmu-dianying")
# 按文件用途统一管理输出目录，全部位于固定项目根目录下。
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "figures"
AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
              45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATION_LABELS = {
    0: "other/unspecified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist",
    16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer",
}
SCHEMAS = {
    "ratings": ["user_id", "movie_id", "rating", "timestamp"],
    "movies": ["movie_id", "title", "genres"],
    "users": ["user_id", "gender", "age", "occupation", "zip_code"],
}


def find_input_dir():
    """只在指定项目内查找，不移动原始数据、不下载数据。"""
    for folder in (ROOT, ROOT / "ml-1m"):
        if all((folder / f"{name}.dat").is_file() for name in SCHEMAS):
            return folder
    raise FileNotFoundError(f"请在 {ROOT} 或其 ml-1m 子目录放齐三份 dat 文件")


def read_dat(folder, name):
    """:: 是多字符分隔符，使用 Python 引擎；Latin-1 兼容电影标题字符。

    先按字符串读取，保留邮编前导零，再显式校验数值。
    不静默跳过字段数异常的行，源文件格式有误时立即报错。
    """
    columns = SCHEMAS[name]
    df = pd.read_csv(
        folder / f"{name}.dat", sep="::", engine="python", header=None,
        dtype="string", encoding="latin-1", keep_default_na=False,
        on_bad_lines="error",
    )
    if len(df.columns) != len(columns):
        raise ValueError(f"{name}.dat 应有 {len(columns)} 列，实际 {len(df.columns)} 列")
    df.columns = columns
    for col in columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    return df


def integer_column(series, minimum, maximum):
    """返回数值列及有效掩码；禁止把 1.5 之类的值直接截断成整数。"""
    values = pd.to_numeric(series, errors="coerce")
    valid = (values.notna() & values.between(minimum, maximum)
             & values.mod(1).eq(0)).fillna(False)
    return values, valid


def clean_table(df, name):
    """删除完全重复行和无效行；每类删除数量记录到质量报告。"""
    stats = {"raw_rows": len(df)}
    df = df.drop_duplicates().copy()
    stats["exact_duplicate_rows_removed"] = stats["raw_rows"] - len(df)
    valid = df.notna().all(axis=1)
    int_specs = {"user_id": (1, 2**31 - 1), "movie_id": (1, 2**31 - 1),
                 "rating": (1, 5), "timestamp": (1, 253402300799),
                 "age": (1, 56), "occupation": (0, 20)}
    for col, (low, high) in int_specs.items():
        if col in df.columns:
            values, col_valid = integer_column(df[col], low, high)
            valid &= col_valid
            # 对非法位置填零只用于安全转换；其所在行稍后统一删除。
            df[col] = values.where(col_valid, 0).astype("int64")
    if name == "users":
        valid &= df["gender"].isin(["M", "F"])
        valid &= df["age"].isin(AGE_LABELS)
        valid &= df["occupation"].isin(OCCUPATION_LABELS)
    if name == "movies":
        # 校验多值字段的基本结构，但保留字符串，不展开成多行。
        valid &= df["genres"].str.split("|", regex=False).map(
            lambda parts: isinstance(parts, list) and all(p.strip() for p in parts)
        )
    if name == "ratings":
        dates = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        valid &= dates.notna()
    stats["invalid_rows_removed"] = int((~valid).sum())
    df = df.loc[valid].copy()
    # 类型标准化可能让 '01' 与 '1' 等源记录变成同一条，再去重一次。
    before = len(df)
    df = df.drop_duplicates().copy()
    stats["normalized_duplicate_rows_removed"] = before - len(df)

    if name in ("movies", "users"):
        key = "movie_id" if name == "movies" else "user_id"
        # 同一主键对应不同元数据时不能随意保留首行，更不能产生多对多连接。
        if df[key].duplicated().any():
            examples = df.loc[df[key].duplicated(False), key].unique()[:5].tolist()
            raise ValueError(f"{name} 存在冲突主键 {examples}，请核查源文件")
        if name == "users":
            df["age_code"] = df["age"].astype("int16")
            df["occupation_code"] = df["occupation"].astype("int16")
            df["age"] = pd.Categorical(
                df["age_code"].map(AGE_LABELS), categories=list(AGE_LABELS.values())
            )
            df["occupation"] = pd.Categorical(
                df["occupation_code"].map(OCCUPATION_LABELS),
                categories=list(OCCUPATION_LABELS.values()),
            )
            df["gender"] = df["gender"].astype("category")
        df = df.sort_values(key, kind="stable")
    else:
        event_key = ["user_id", "movie_id", "timestamp"]
        if df.duplicated(event_key).any():
            raise ValueError("同一用户、电影、时间戳存在不同评分，请核查源文件")
        # 相同用户-电影在不同时间出现时保留全部事件，避免未来记录覆盖历史。
        stats["repeated_user_movie_rows_preserved"] = int(
            df.duplicated(["user_id", "movie_id"]).sum()
        )
        df["rating"] = df["rating"].astype("int8")
        df["rating_datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("timestamp", kind="stable")
    stats["clean_rows"] = len(df)
    return df.reset_index(drop=True), stats


def plot_quality(ratings, movies, users, output_dir):
    """描述性 EDA；全量电影交互数不能直接用于后续训练或长尾分组。"""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ratings["rating"].value_counts().reindex(range(1, 6), fill_value=0).plot.bar(
        ax=axes[0, 0], color="#3977ae", rot=0, title="Rating distribution")
    axes[0, 0].set(xlabel="Rating", ylabel="Events")
    monthly = ratings.set_index("rating_datetime").resample("MS").size()
    axes[0, 1].plot(monthly.index, monthly.values, color="#3977ae")
    axes[0, 1].set(title="Monthly rating events (UTC)", ylabel="Events")
    axes[0, 1].tick_params(axis="x", rotation=30)
    counts = ratings.groupby("movie_id").size().reindex(movies["movie_id"], fill_value=0)
    ranked = counts.sort_values(ascending=False).to_numpy()
    axes[0, 2].plot(range(1, len(ranked) + 1), ranked, color="#d07932")
    axes[0, 2].set_ylim(bottom=0)
    axes[0, 2].set(xscale="log", yscale="symlog", xlabel="Movie rank (log)",
                   ylabel="Events (symlog, includes zero)",
                   title="Full-data popularity: EDA only")
    ratings.groupby("user_id").size().plot.hist(bins=50, ax=axes[1, 0], color="#519c91")
    axes[1, 0].set(title="User activity", xlabel="Events per user", ylabel="Users")
    users["age"].value_counts(sort=False).plot.bar(ax=axes[1, 1], color="#519c91", rot=30)
    axes[1, 1].set(title="Age categories", ylabel="Users")
    movies["genres"].str.split("|", regex=False).explode().value_counts().plot.bar(
        ax=axes[1, 2], color="#8b78a5", rot=65, title="Genres (multi-label)")
    axes[1, 2].set_ylabel("Movies")
    fig.suptitle("MovieLens-1M | Stage 1 data quality", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "data1_quality.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(export_csv=False):
    for folder in (DATA_DIR, REPORT_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    input_dir = find_input_dir()
    tables, report = {}, {"input_dir": str(input_dir), "tables": {}}
    for name in SCHEMAS:
        tables[name], report["tables"][name] = clean_table(read_dat(input_dir, name), name)
    ratings, movies, users = (tables[n] for n in ("ratings", "movies", "users"))
    # 外键校验：不使用 inner join 静默丢掉无法关联的评分。
    missing_users = ~ratings["user_id"].isin(users["user_id"])
    missing_movies = ~ratings["movie_id"].isin(movies["movie_id"])
    report["orphan_rating_rows"] = {
        "missing_user": int(missing_users.sum()), "missing_movie": int(missing_movies.sum())}
    if missing_users.any() or missing_movies.any():
        raise ValueError(f"评分有未匹配的元数据：{report['orphan_rating_rows']}，请核查源文件")
    wide = ratings.merge(users, on="user_id", how="left", validate="many_to_one", sort=False)
    wide = wide.merge(movies, on="movie_id", how="left", validate="many_to_one", sort=False)
    wide = wide.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(wide) != len(ratings) or wide.isna().any().any():
        raise RuntimeError("宽表行数或非空校验失败")
    if ratings.empty or movies.empty or users.empty:
        raise ValueError("清洗后存在空表，无法继续")
    report["wide_rows"] = len(wide)
    report["wide_columns"] = wide.columns.tolist()
    report["time_range_utc"] = [str(ratings["rating_datetime"].min()),
                                str(ratings["rating_datetime"].max())]
    report["unrated_movies"] = int((~movies["movie_id"].isin(ratings["movie_id"])).sum())
    report["rating_distribution"] = {
        str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()}
    report["notes"] = [
        "No split, CTR labels, negative sampling or fitted features in stage 1.",
        "Full-data popularity plot is EDA only; define tail groups using training data later.",
        "Timestamp ties retain source order; this order does not imply event causality.",
        "User profiles have no historical timestamps; snapshot availability is an assumption.",
    ]
    for name, df in {"df_ratings": ratings, "df_movies": movies,
                     "df_users": users, "df_wide": wide}.items():
        # pickle 保留 category、字符串和 UTC 时间类型；仅加载可信的本地文件。
        df.to_pickle(DATA_DIR / f"{name}.pkl")
        if export_csv:
            df.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    plot_quality(ratings, movies, users, FIGURE_DIR)
    with (REPORT_DIR / "data1_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n完成：数据表：{DATA_DIR}；质量报告：{REPORT_DIR}；图表：{FIGURE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-csv", action="store_true", help="额外输出四份 CSV")
    main(export_csv=parser.parse_args().export_csv)


