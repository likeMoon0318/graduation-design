# -*- coding: utf-8 -*-
# 数据清洗与数据质量报告
# =====================================================================
# 输入：data/raw/orders_raw.csv（含缺失、异常、重复、格式混乱）
# 输出：data/processed/orders_clean.csv
#       results/tables/data_quality_report.csv   数据质量问题诊断表
#       results/tables/cleaning_log.json         清洗动作日志（可复现凭证）
#
# 清洗原则：
#   1. 能修则修、不能修则标记，绝不整行删除（避免人为损失样本）
#   2. 优先利用字段之间的业务勾稽关系修复，而不是简单填平均值
#      例如：数量 = (实付金额 + 优惠) / 单价
#   3. 每一步都记录处理条数，供报告做"清洗前后对照"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_FILE, CLEAN_FILE, QUALITY_FILE, TABLE_DIR  # noqa: E402


# ---------------------------------------------------------------------
# 一、读取与初步诊断
# ---------------------------------------------------------------------

def load_raw():
    print(f"[读取] {RAW_FILE.name}")
    df = pd.read_csv(RAW_FILE, dtype={"order_id": str, "customer_id": str})
    print(f"[读取] 原始数据形状：{df.shape[0]:,} 行 x {df.shape[1]} 列")
    return df


def _blank_mask(df, col):
    """缺失值判定：既包括 NaN，也包括空字符串与纯空格。"""
    s = df[col]
    return s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.strip() == "nan")


def diagnose(df):
    """对原始数据做全面体检，返回问题清单（用于生成质量报告）。"""
    issues = []
    n = len(df)

    # 1) 缺失值（promotion 为空是业务含义"非大促"，不计为缺陷）
    for col in df.columns:
        if col == "promotion":
            continue
        miss = int(_blank_mask(df, col).sum())
        if miss > 0:
            issues.append({
                "检查项": "缺失值", "字段": col, "问题数量": miss,
                "占比": f"{miss / n:.2%}", "处理方式": "按字段语义分别处理（见清洗日志）",
            })
    promo_blank = int(_blank_mask(df, "promotion").sum())
    issues.append({"检查项": "业务空值", "字段": "promotion", "问题数量": promo_blank,
                   "占比": f"{promo_blank / n:.2%}", "处理方式": "空值表示非大促订单，填充为「非大促」"})

    # 2) 重复记录
    dup_full = int(df.duplicated().sum())
    dup_id = int(df.duplicated(subset=["order_id"], keep="first").sum())
    issues.append({"检查项": "重复值", "字段": "全部字段", "问题数量": dup_full,
                   "占比": f"{dup_full / n:.2%}", "处理方式": "删除完全重复的行"})
    issues.append({"检查项": "重复值", "字段": "order_id", "问题数量": dup_id,
                   "占比": f"{dup_id / n:.2%}", "处理方式": "按订单号去重，保留首次出现记录"})

    # 3) 金额格式异常（含货币符号或千分位，无法直接转数值）
    amount_str = df["amount"].astype(str)
    bad_amount = int(amount_str.str.contains("¥", regex=False).sum())
    issues.append({"检查项": "格式异常", "字段": "amount", "问题数量": bad_amount,
                   "占比": f"{bad_amount / n:.2%}", "处理方式": "剥离货币符号与千分位后转数值"})

    # 4) 时间格式异常（同一字段混用 yyyy-mm-dd 与 yyyy/mm/dd）
    raw_time = df["order_time"].astype(str)
    slash = int(raw_time.str.contains("/", regex=False).sum())
    issues.append({"检查项": "格式异常", "字段": "order_time", "问题数量": slash,
                   "占比": f"{slash / n:.2%}", "处理方式": "统一解析为 datetime 类型"})

    # 5) 逻辑异常：数量为负
    qty = pd.to_numeric(df["quantity"], errors="coerce")
    neg_qty = int((qty < 0).sum())
    issues.append({"检查项": "逻辑异常", "字段": "quantity", "问题数量": neg_qty,
                   "占比": f"{neg_qty / n:.2%}", "处理方式": "视为录入错误，按金额与单价反推数量"})

    # 6) 逻辑异常：金额与「单价 x 数量 - 折扣」严重不符（含异常放大/缩小的金额）
    amount_num = pd.to_numeric(
        amount_str.str.replace("¥", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    expect = df["unit_price"] * qty.clip(lower=1) - df["discount"]
    deviation = (amount_num - expect).abs() / expect.replace(0, np.nan)
    mismatch = int((deviation > 0.05).sum())
    issues.append({"检查项": "逻辑异常", "字段": "amount", "问题数量": mismatch,
                   "占比": f"{mismatch / n:.2%}", "处理方式": "按 单价x数量-折扣 重算修正"})

    # 7) 文本不规范
    for col in ("channel", "category"):
        s = df[col].astype(str)
        space = s != s.str.strip()                                   # 首尾含空白字符
        case = s.str.strip().isin(["app", "web", "mini"])            # 英文大小写不统一
        bad = int((space | case).sum())
        issues.append({"检查项": "文本不规范", "字段": col, "问题数量": bad,
                       "占比": f"{bad / n:.2%}",
                       "处理方式": "去除首尾空白字符并统一大小写"})

    return pd.DataFrame(issues)


# ---------------------------------------------------------------------
# 二、清洗流程
# ---------------------------------------------------------------------

def clean(df):
    log = {}
    before = len(df)

    # ---------- 步骤 1：删除完全重复行 ----------
    df = df.drop_duplicates()
    log["删除完全重复行"] = before - len(df)
    print(f"[清洗 01] 删除完全重复行：{log['删除完全重复行']} 行，剩余 {len(df):,} 行")

    # ---------- 步骤 2：按订单号去重 ----------
    step = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    log["按订单号去重"] = step - len(df)
    print(f"[清洗 02] 按订单号去重：{log['按订单号去重']} 行，剩余 {len(df):,} 行")

    # ---------- 步骤 3：文本字段规范化 ----------
    ws_fix = 0
    for col in ("channel", "category", "province", "city"):
        s = df[col].fillna("").astype(str)
        ws_fix += int((s != s.str.strip()).sum())
        df[col] = s.str.strip()
    # 英文大小写统一：小写渠道名映射回标准写法
    df["channel"] = df["channel"].replace({"app": "APP", "web": "网页", "mini": "小程序"})
    log["文本规范化"] = ws_fix
    print(f"[清洗 03] 文本字段规范化：{ws_fix} 条去除首尾空白，渠道大小写已统一")

    # ---------- 步骤 4：金额文本解析 ----------
    amount_str = df["amount"].astype(str)
    log["金额格式修复"] = int(amount_str.str.contains("¥", regex=False).sum())
    df["amount"] = pd.to_numeric(
        amount_str.str.replace("¥", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce")
    print(f"[清洗 04] 金额格式修复：{log['金额格式修复']} 条文本型金额转为数值")

    # ---------- 步骤 5：数量修复（优先用金额勾稽关系反推） ----------
    qty = pd.to_numeric(df["quantity"], errors="coerce")
    neg = int((qty < 0).sum())
    invalid = qty.isna() | (qty <= 0)
    log["数量异常条数"] = int(invalid.sum())
    log["数量负值条数"] = neg
    # 由「实付金额 = 单价 x 数量 - 折扣」反推数量：数量 = (金额 + 折扣) / 单价
    derived = ((df["amount"] + df["discount"]) / df["unit_price"]).round().clip(1, 10)
    qty_fixed = qty.mask(invalid, derived)
    derived_ok = int(qty_fixed[invalid].notna().sum())
    # 仍无法反推的（金额异常导致），按品类中位数填补
    median_by_cat = qty_fixed.groupby(df["category"]).transform("median")
    still_bad = qty_fixed.isna() | (qty_fixed <= 0)
    qty_fixed = qty_fixed.mask(still_bad, median_by_cat).fillna(qty_fixed.median())
    df["quantity"] = qty_fixed.round().astype(int).clip(1, 10)
    log["数量勾稽反推"] = derived_ok
    log["数量中位数填补"] = int(still_bad.sum())
    print(f"[清洗 05] 数量修复：异常 {log['数量异常条数']} 条（负值 {neg} 条），"
          f"其中 {derived_ok} 条由金额÷单价反推，{log['数量中位数填补']} 条按品类中位数填补")

    # ---------- 步骤 6：金额逻辑校验与重算 ----------
    expect = (df["unit_price"] * df["quantity"] - df["discount"]).round(2)
    deviation = (df["amount"] - expect).abs() / expect.replace(0, np.nan)
    bad = (deviation > 0.05) | df["amount"].isna() | (df["amount"] <= 0)
    log["金额逻辑修正"] = int(bad.sum())
    df.loc[bad, "amount"] = expect[bad]
    df["amount"] = df["amount"].round(2)
    print(f"[清洗 06] 金额逻辑修正：{log['金额逻辑修正']} 条异常金额按「单价x数量-折扣」重算")

    # ---------- 步骤 7：折扣越界修正 ----------
    gross = df["unit_price"] * df["quantity"]
    over = df["discount"] > gross
    log["折扣越界修正"] = int(over.sum())
    df.loc[over, "discount"] = (gross[over] * 0.5).round(2)
    print(f"[清洗 07] 折扣越界修正：{log['折扣越界修正']} 条（优惠金额高于商品总价）")

    # ---------- 步骤 8：时间字段统一解析 ----------
    t = pd.to_datetime(df["order_time"], errors="coerce", format="mixed")
    log["时间解析失败"] = int(t.isna().sum())
    df["order_time"] = t
    print(f"[清洗 08] 时间字段统一解析完成，解析失败 {log['时间解析失败']} 条")

    # ---------- 步骤 9：地区缺失标记 ----------
    log["地区缺失标记"] = int((df["province"] == "").sum())
    df["province"] = df["province"].replace("", "未知")
    df["city"] = df["city"].replace("", "未知")
    print(f"[清洗 09] 地区缺失标记为「未知」：{log['地区缺失标记']} 条（保留样本，单列为一类）")

    # ---------- 步骤 10：大促字段语义化 ----------
    df["promotion"] = df["promotion"].replace("", "非大促").fillna("非大促")
    df["is_promo"] = df["promotion"] != "非大促"
    log["大促字段语义化"] = int(df["is_promo"].sum())
    print(f"[清洗 10] 大促字段语义化：识别大促订单 {log['大促字段语义化']:,} 条（占比 "
          f"{df['is_promo'].mean():.2%}）")

    # ---------- 步骤 11：评分缺失保留 + 生成建模填补列 ----------
    log["评分缺失保留"] = int(df["rating"].isna().sum())
    df["rating_filled"] = df.groupby("category")["rating"].transform(lambda s: s.fillna(s.median()))
    print(f"[清洗 11] 评分缺失 {log['评分缺失保留']:,} 条（{df['rating'].isna().mean():.2%}）："
          f"分析时保留缺失，另生成 rating_filled 供建模使用")

    # ---------- 步骤 12：按品类标记离群订单金额 ----------
    # 不同品类价格量级差异大，因此分品类在对数尺度上做 1.5 倍四分位距判定。
    # 注意：这里的离群是"业务上真实存在的贵价订单/多件订单"，不是数据错误，
    #       仅用于箱线图等图表中标注，绝不删除，避免破坏真实的销售分布。
    log_amount = np.log10(df["amount"].clip(lower=0.01))
    q1 = log_amount.groupby(df["category"]).transform(lambda s: s.quantile(0.25))
    q3 = log_amount.groupby(df["category"]).transform(lambda s: s.quantile(0.75))
    fence = q3 + 1.5 * (q3 - q1)
    df["is_amount_outlier"] = log_amount > fence
    log["金额离群标记"] = int(df["is_amount_outlier"].sum())
    print(f"[清洗 12] 分品类离群标记：{log['金额离群标记']:,} 条"
          f"（占比 {df['is_amount_outlier'].mean():.2%}，判定为正常高价订单，不删除）")

    log["清洗前行数"] = int(before)
    log["清洗后行数"] = int(len(df))
    return df, log


# ---------------------------------------------------------------------
# 三、清洗后校验
# ---------------------------------------------------------------------

def verify(df):
    checks = {
        "完全重复行": int(df.duplicated().sum()),
        "订单号重复": int(df.duplicated(subset=["order_id"]).sum()),
        "金额缺失": int(df["amount"].isna().sum()),
        "金额非正数": int((df["amount"] <= 0).sum()),
        "数量非正数": int((df["quantity"] <= 0).sum()),
        "时间解析失败": int(df["order_time"].isna().sum()),
        "渠道带空格": int(df["channel"].ne(df["channel"].str.strip()).sum()),
        "品类带空格": int(df["category"].ne(df["category"].str.strip()).sum()),
        "地区为空": int((df["province"] == "").sum()),
        "金额勾稽不符": int((((df["amount"] - (df["unit_price"] * df["quantity"] - df["discount"])).abs()
                            / (df["unit_price"] * df["quantity"] - df["discount"]).replace(0, np.nan)) > 0.05).sum()),
    }
    print("\n[校验] 清洗后数据质量复查：")
    for k, v in checks.items():
        print(f"    {k:<12} {v:>8,}   {'通过' if v == 0 else '需关注'}")
    return checks


def main():
    df_raw = load_raw()
    report = diagnose(df_raw)

    print("\n" + "=" * 70)
    print("一、数据质量问题诊断")
    print("=" * 70)
    print(report.to_string(index=False))

    print("\n" + "=" * 70)
    print("二、执行清洗")
    print("=" * 70)
    df_clean, log = clean(df_raw)
    checks = verify(df_clean)

    print("\n" + "=" * 70)
    print("三、输出结果")
    print("=" * 70)
    df_clean.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
    print(f"清洗后数据：{CLEAN_FILE}（{len(df_clean):,} 行 x {df_clean.shape[1]} 列）")

    report.to_csv(QUALITY_FILE, index=False, encoding="utf-8-sig")
    print(f"质量诊断表：{QUALITY_FILE}")

    with open(TABLE_DIR / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump({"清洗动作": log, "清洗后校验": checks}, f, ensure_ascii=False, indent=2)
    print(f"清洗日志：{TABLE_DIR / 'cleaning_log.json'}")
    return df_clean


if __name__ == "__main__":
    main()
