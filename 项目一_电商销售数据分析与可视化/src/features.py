# -*- coding: utf-8 -*-
# 特征工程：时间特征、客户 RFM 指标、日粒度汇总表
# =====================================================================
# 输入：data/processed/orders_clean.csv
# 输出：data/processed/customer_rfm.csv     客户 RFM 明细（供分群使用）
#       data/processed/daily_stats.csv      日粒度经营指标（供看板快速读取）

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEAN_FILE, CUSTOMER_FILE, PROCESSED_DIR  # noqa: E402

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 下单时段划分（用于分析用户的作息规律）
TIME_SLOT_BINS = [-1, 6, 11, 14, 18, 23]
TIME_SLOT_LABELS = ["凌晨(0-6)", "上午(7-11)", "午间(12-14)", "下午(15-18)", "晚间(19-23)"]


def load_clean():
    """读取清洗后的订单数据。"""
    df = pd.read_csv(CLEAN_FILE, dtype={"order_id": str, "customer_id": str},
                     parse_dates=["order_time"])
    return df


def add_time_features(df):
    """派生时间维度特征。"""
    t = df["order_time"]
    df["date"] = t.dt.date
    df["year_month"] = t.dt.to_period("M").astype(str)
    df["month"] = t.dt.month
    df["month_name"] = t.dt.month.astype(str) + "月"
    df["quarter"] = t.dt.quarter
    df["week"] = t.dt.isocalendar().week.astype(int)
    df["weekday"] = t.dt.weekday
    df["weekday_name"] = df["weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    df["hour"] = t.dt.hour
    df["is_weekend"] = df["weekday"] >= 5
    df["time_slot"] = pd.cut(df["hour"], bins=TIME_SLOT_BINS, labels=TIME_SLOT_LABELS)
    df["day_of_year"] = t.dt.dayofyear
    return df


def build_customer_rfm(df):
    """构造客户 RFM 指标。

    R：最近一次购买距观测期末的天数（越小越活跃）
    F：观测期内下单次数
    M：观测期内消费总金额
    """
    snapshot = df["order_time"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        最近购买时间=("order_time", "max"),
        订单数=("order_id", "count"),
        消费总额=("amount", "sum"),
        平均客单价=("amount", "mean"),
        退货次数=("is_return", "sum"),
    ).reset_index()
    rfm["R"] = (snapshot - rfm["最近购买时间"]).dt.days
    rfm["F"] = rfm["订单数"]
    rfm["M"] = rfm["消费总额"].round(2)
    rfm["退货率"] = (rfm["退货次数"] / rfm["订单数"]).round(4)

    # 附加客户属性（取该客户最常出现的取值）
    attr = (df.groupby("customer_id")[["member_level", "province"]]
            .agg(lambda s: s.mode().iat[0]).reset_index()
            .rename(columns={"member_level": "会员等级", "province": "省份"}))
    rfm = rfm.merge(attr, on="customer_id", how="left")

    # RFM 三维打分（1—5 分，用于业务解读；分群本身使用原始值）
    rfm["R_Score"] = pd.qcut(rfm["R"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["F"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["M"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_总分"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    # 依据 RFM 总分给出客户价值标签（业务解释用）
    def label(score):
        if score >= 13:
            return "重要价值客户"
        if score >= 10:
            return "潜力客户"
        if score >= 7:
            return "一般维持客户"
        return "流失预警客户"

    rfm["价值标签"] = rfm["RFM_总分"].map(label)
    return rfm


def build_daily_stats(df):
    """日粒度经营指标，供看板快速读取。"""
    daily = df.groupby("date").agg(
        订单量=("order_id", "count"),
        销售额=("amount", "sum"),
        活跃客户数=("customer_id", "nunique"),
        客单价=("amount", "mean"),
        退货率=("is_return", "mean"),
    ).reset_index()
    daily["销售额"] = daily["销售额"].round(2)
    daily["客单价"] = daily["客单价"].round(2)
    promo = df.groupby("date")["promotion"].agg(lambda s: s.mode().iat[0]).reset_index()
    daily = daily.merge(promo, on="date", how="left")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["7日移动平均"] = daily["销售额"].rolling(7, min_periods=1).mean().round(2)
    return daily


def main():
    print("[读取] 清洗后的订单数据")
    df = load_clean()
    df = add_time_features(df)
    print(f"        {len(df):,} 行 x {df.shape[1]} 列")

    rfm = build_customer_rfm(df)
    rfm.to_csv(CUSTOMER_FILE, index=False, encoding="utf-8-sig")
    print(f"[输出] 客户 RFM 明细：{CUSTOMER_FILE.name}（{len(rfm):,} 位客户）")
    print("\n客户价值分布：")
    print(rfm["价值标签"].value_counts().to_string())

    daily = build_daily_stats(df)
    daily.to_csv(PROCESSED_DIR / "daily_stats.csv", index=False, encoding="utf-8-sig")
    print(f"\n[输出] 日粒度指标：daily_stats.csv（{len(daily)} 天）")
    print(f"       全年销售额 {daily['销售额'].sum() / 1e8:.2f} 亿元，"
          f"日均 {daily['销售额'].mean() / 1e4:.1f} 万元")

    # 关键派生指标，供报告引用
    print("\n[核心指标]")
    print(f"        总订单量      {len(df):,} 单")
    print(f"        总销售额      {df['amount'].sum() / 1e8:.2f} 亿元")
    print(f"        平均客单价    {df['amount'].mean():.2f} 元")
    print(f"        整体退货率    {df['is_return'].mean():.2%}")
    print(f"        消费客户数    {df['customer_id'].nunique():,} 人")
    return df, rfm


if __name__ == "__main__":
    main()
