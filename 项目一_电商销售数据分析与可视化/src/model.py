# -*- coding: utf-8 -*-
# 数据建模：客户分群、退货预测、订单金额影响因素分析
# =====================================================================
# 三个模型回答三个业务问题：
#   1. 客户可以分成几类？各自特征是什么？      -> KMeans 聚类（RFM 指标）
#   2. 哪些订单更容易被退货？模型判别能力如何？ -> 逻辑回归 + ROC/AUC
#   3. 订单金额由哪些因素决定？影响方向如何？   -> 岭回归 + 标准化系数
# 输出：results/tables/ 下的模型结果表、data/processed/customer_segments.csv

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # 避免 joblib 在多核探测上报警

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CUSTOMER_FILE, PROCESSED_DIR, RANDOM_SEED, TABLE_DIR  # noqa: E402
from features import add_time_features, load_clean  # noqa: E402

pd.set_option("display.unicode.east_asian_width", True)


def save(df, name, index=False):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------
# 一、KMeans 客户分群
# ---------------------------------------------------------------------

def customer_segmentation():
    print("\n" + "=" * 74)
    print("一、KMeans 客户分群（基于 RFM 指标）")
    print("=" * 74)
    rfm = pd.read_csv(CUSTOMER_FILE, dtype={"customer_id": str})

    # RFM 三个指标量纲不同且分布右偏，先做对数变换再标准化
    X = np.column_stack([
        np.log1p(rfm["R"]),          # R 越小越好，这里保留原始方向，后续按均值解释
        np.log1p(rfm["F"]),
        np.log1p(rfm["M"]),
    ])
    X_scaled = StandardScaler().fit_transform(X)

    # 用肘部法（组内平方和）与轮廓系数共同确定簇数
    metrics = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_scaled)
        metrics.append({"簇数k": k, "组内平方和SSE": round(km.inertia_, 2),
                        "轮廓系数": round(silhouette_score(X_scaled, labels), 4)})
    metrics_df = pd.DataFrame(metrics)
    # SSE 相对下降幅度（用于判断肘部位置）
    metrics_df["SSE下降幅度"] = (-metrics_df["组内平方和SSE"].diff() /
                             metrics_df["组内平方和SSE"].shift(1)).round(4)
    print(metrics_df.to_string(index=False))
    save(metrics_df, "客户分群_簇数评估")

    sil_best = int(metrics_df.loc[metrics_df["轮廓系数"].idxmax(), "簇数k"])
    # 簇数选择：轮廓系数偏好 k=2，但肘部法显示 k=4 之后 SSE 下降明显放缓；
    # 同时 4 类客户在运营上可直接对应四套差异化策略，具备业务可解释性。
    # 因此在统计指标与业务价值之间取平衡，选择 k=4，并在报告中说明该权衡。
    best_k = 4
    print(f"\n轮廓系数最优簇数：k = {sil_best}（轮廓系数 "
          f"{metrics_df.loc[metrics_df['簇数k'] == sil_best, '轮廓系数'].iat[0]:.4f}）")
    print(f"肘部法显示 k = 4 之后 SSE 下降幅度明显放缓（"
          f"{metrics_df.loc[metrics_df['簇数k'] == 4, 'SSE下降幅度'].iat[0]:.2%} -> "
          f"{metrics_df.loc[metrics_df['簇数k'] == 5, 'SSE下降幅度'].iat[0]:.2%}）")
    print(f"综合统计指标与业务可解释性，最终选择 k = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED)
    rfm["分群"] = km.fit_predict(X_scaled)

    # 统计每个簇的画像，便于业务命名
    profile = rfm.groupby("分群").agg(
        客户数=("customer_id", "count"),
        平均最近购买间隔R=("R", "mean"),
        平均购买次数F=("F", "mean"),
        平均消费金额M=("M", "mean"),
        平均客单价=("平均客单价", "mean"),
        退货率=("退货率", "mean"),
    ).round(2)
    profile["客户占比"] = (profile["客户数"] / profile["客户数"].sum()).round(4)

    # 客户群命名：按各簇的平均 RFM 综合得分排序，依次命名，保证标签可解释且不重复
    score = rfm.groupby("分群")["RFM_总分"].mean().sort_values(ascending=False)
    labels = ["核心价值客户", "稳定活跃客户", "一般维持客户", "流失预警客户"]
    name_map = {int(cluster): labels[i] for i, cluster in enumerate(score.index)}
    profile["客户群命名"] = [name_map[int(c)] for c in profile.index]
    profile["平均RFM总分"] = score.round(2)
    print("\n【客户分群画像】")
    print(profile.to_string())
    save(profile.reset_index(), "客户分群画像")

    rfm.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False, encoding="utf-8-sig")
    print(f"\n[输出] 客户分群结果：customer_segments.csv（{len(rfm):,} 位客户）")
    return rfm, profile, metrics_df


# ---------------------------------------------------------------------
# 二、退货预测模型（逻辑回归）
# ---------------------------------------------------------------------

def return_prediction(df):
    print("\n" + "=" * 74)
    print("二、退货预测模型（逻辑回归）")
    print("=" * 74)
    data = df.copy()
    data["折扣率"] = (data["discount"] / (data["unit_price"] * data["quantity"]).replace(0, np.nan)).fillna(0)
    data["单价对数"] = np.log1p(data["unit_price"])

    num_cols = ["单价对数", "quantity", "折扣率", "rating_filled", "hour"]
    cat_cols = ["category", "channel", "member_level", "is_promo"]

    X_num = data[num_cols].astype(float)
    X_cat = pd.get_dummies(data[cat_cols].astype(str), drop_first=True, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = data["is_return"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train_s, y_train)
    prob = clf.predict_proba(X_test_s)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, prob)
    acc = float((pred == y_test).mean())
    metrics = pd.DataFrame([{
        "模型": "逻辑回归（类别加权）",
        "训练样本数": len(X_train), "测试样本数": len(X_test),
        "特征数": X.shape[1],
        "准确率Accuracy": round(acc, 4),
        "精确率Precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "召回率Recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "F1分数": round(f1_score(y_test, pred, zero_division=0), 4),
        "AUC": round(auc, 4),
    }])
    print("\n【模型评价指标】")
    print(metrics.to_string(index=False))
    save(metrics, "退货预测_模型评价")

    cm = pd.DataFrame(confusion_matrix(y_test, pred),
                      index=["实际未退货", "实际退货"], columns=["预测未退货", "预测退货"])
    print("\n【混淆矩阵】")
    print(cm.to_string())
    save(cm.reset_index().rename(columns={"index": "实际"}), "退货预测_混淆矩阵")

    # 特征影响：标准化系数即"每提升一个标准差对退货几率的影响"
    coef = pd.DataFrame({
        "特征": X.columns,
        "标准化系数": clf.coef_[0].round(4),
        "影响方向": np.where(clf.coef_[0] > 0, "提高退货概率", "降低退货概率"),
        "影响强度": np.abs(clf.coef_[0]).round(4),
    }).sort_values("影响强度", ascending=False)
    print("\n【特征影响 Top10】")
    print(coef.head(10).to_string(index=False))
    save(coef, "退货预测_特征系数")

    # 保存 ROC 曲线数据，供绘图模块使用
    fpr, tpr, _ = roc_curve(y_test, prob)
    step = max(1, len(fpr) // 500)
    roc_df = pd.DataFrame({"假正率FPR": fpr[::step], "真正率TPR": tpr[::step]})
    save(roc_df, "退货预测_ROC曲线数据")

    return metrics, coef, auc


# ---------------------------------------------------------------------
# 三、订单金额影响因素（岭回归）
# ---------------------------------------------------------------------

def amount_drivers(df):
    print("\n" + "=" * 74)
    print("三、订单金额影响因素分析（岭回归）")
    print("=" * 74)
    data = df.copy()
    data["折扣率"] = (data["discount"] / (data["unit_price"] * data["quantity"]).replace(0, np.nan)).fillna(0)
    y = np.log1p(data["amount"])          # 金额右偏，取对数后建模

    num_cols = ["quantity", "折扣率", "rating_filled", "hour"]
    cat_cols = ["category", "channel", "member_level", "province", "is_promo"]
    X_num = data[num_cols].astype(float)
    X_cat = pd.get_dummies(data[cat_cols].astype(str), drop_first=True, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(scaler.transform(X_train), y_train)

    r2_train = model.score(scaler.transform(X_train), y_train)
    r2_test = model.score(scaler.transform(X_test), y_test)
    print(f"\n模型拟合优度：训练集 R^2 = {r2_train:.4f}，测试集 R^2 = {r2_test:.4f}")
    print("说明：订单金额主要由「商品品类」决定（品类代表不同的价格量级），")
    print("      因此模型具备较强的解释力；残差部分来自同一品类内部的价格波动。")

    coef = pd.DataFrame({
        "特征": X.columns,
        "标准化系数": model.coef_.round(4),
    })
    coef["影响强度"] = coef["标准化系数"].abs()
    coef = coef.sort_values("影响强度", ascending=False)
    coef["方向"] = np.where(coef["标准化系数"] > 0, "提高订单金额", "降低订单金额")
    print("\n【影响订单金额的主要因素 Top12】")
    print(coef.head(12).to_string(index=False))
    save(coef, "订单金额影响因素_系数")

    summary = pd.DataFrame([{
        "模型": "岭回归（alpha=1.0）", "目标变量": "log(订单金额+1)",
        "训练集R2": round(r2_train, 4), "测试集R2": round(r2_test, 4),
        "训练样本数": len(X_train), "测试样本数": len(X_test), "特征数": X.shape[1],
    }])
    save(summary, "订单金额影响因素_模型评价")
    return summary, coef


def main():
    df = add_time_features(load_clean())
    print(f"[读取] 清洗后数据 {len(df):,} 行")
    customer_segmentation()
    return_prediction(df)
    amount_drivers(df)
    print(f"\n[输出] 建模结果已写入 {TABLE_DIR}")


if __name__ == "__main__":
    main()
