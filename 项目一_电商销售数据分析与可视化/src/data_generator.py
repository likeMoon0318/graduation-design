# -*- coding: utf-8 -*-
# 电商订单数据仿真生成器
# =====================================================================
# 生成一份贴近真实的 2025 年全年电商订单明细数据。数据的"真实感"来自三层：
#   1. 业务机理：订单量 = 基线 x 年度趋势 x 周周期 x 季节波动 x 节假日脉冲 x 随机扰动
#   2. 变量关联：客户等级影响客单价、品类影响退货率、评分影响是否退货、大促影响折扣
#   3. 已知脏数据：按固定比例注入缺失、异常、重复、格式混乱，并记录注入清单，
#      便于后续清洗环节做"清洗前后对照"，让数据质量分析有据可依
#
# 运行：python src/data_generator.py
# 输出：data/raw/orders_raw.csv、data/raw/injection_log.json

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_FILE, RAW_DIR, RANDOM_SEED  # noqa: E402

# =====================================================================
# 一、业务常量定义
# =====================================================================

# 品类：价格对数均值、对数标准差、销售占比、基准退货率、基准评分
CATEGORIES = {
    "手机数码": dict(price_mu=7.78, price_sigma=0.62, weight=0.10, return_rate=0.045, rating_mu=4.55,
                     products=["智能手机", "蓝牙耳机", "智能手表", "平板电脑", "移动电源", "无线充电器"]),
    "家用电器": dict(price_mu=7.38, price_sigma=0.58, weight=0.08, return_rate=0.035, rating_mu=4.60,
                     products=["空气炸锅", "扫地机器人", "电饭煲", "加湿器", "破壁机", "挂烫机"]),
    "服饰鞋包": dict(price_mu=5.63, price_sigma=0.66, weight=0.26, return_rate=0.185, rating_mu=4.25,
                     products=["冬季羽绒服", "休闲运动鞋", "纯棉T恤", "牛仔裤", "双肩背包", "针织连衣裙"]),
    "美妆个护": dict(price_mu=5.25, price_sigma=0.56, weight=0.16, return_rate=0.050, rating_mu=4.45,
                     products=["保湿面霜", "氨基酸洁面", "精华液", "防晒喷雾", "口红礼盒", "洗发水套装"]),
    "食品生鲜": dict(price_mu=4.44, price_sigma=0.52, weight=0.18, return_rate=0.025, rating_mu=4.35,
                     products=["进口牛肉", "应季水果礼盒", "坚果大礼包", "速冻水饺", "现磨咖啡豆", "低脂牛奶"]),
    "图书文娱": dict(price_mu=4.00, price_sigma=0.45, weight=0.06, return_rate=0.015, rating_mu=4.65,
                     products=["考研数学辅导", "少儿绘本套装", "文学小说", "手账本", "钢笔礼盒"]),
    "母婴玩具": dict(price_mu=5.39, price_sigma=0.60, weight=0.08, return_rate=0.060, rating_mu=4.50,
                     products=["婴儿纸尿裤", "积木益智玩具", "儿童安全座椅", "辅食机", "婴儿推车"]),
    "运动户外": dict(price_mu=5.83, price_sigma=0.62, weight=0.08, return_rate=0.090, rating_mu=4.40,
                     products=["瑜伽垫", "跑步机", "登山背包", "羽毛球拍", "露营帐篷", "健身哑铃"]),
}

# 地区：省、主要城市候选、消费能力系数
REGIONS = [
    ("广东", ["深圳", "广州", "东莞"], 1.28),
    ("北京", ["北京"], 1.30),
    ("上海", ["上海"], 1.26),
    ("浙江", ["杭州", "宁波", "温州"], 1.12),
    ("江苏", ["南京", "苏州", "无锡"], 1.10),
    ("四川", ["成都", "绵阳"], 1.02),
    ("湖北", ["武汉", "宜昌"], 1.00),
    ("福建", ["厦门", "福州"], 0.98),
    ("山东", ["青岛", "济南"], 0.96),
    ("湖南", ["长沙", "株洲"], 0.94),
    ("陕西", ["西安", "咸阳"], 0.93),
    ("河南", ["郑州", "洛阳"], 0.92),
]
REGION_WEIGHTS = np.array([0.135, 0.105, 0.110, 0.095, 0.090, 0.080, 0.075,
                           0.060, 0.065, 0.055, 0.045, 0.085])
REGION_WEIGHTS = REGION_WEIGHTS / REGION_WEIGHTS.sum()

# 渠道：基准占比（网页端份额全年逐步被 APP 蚕食）
CHANNELS = ["APP", "小程序", "网页", "线下门店"]
CHANNEL_BASE = np.array([0.44, 0.27, 0.18, 0.11])
CHANNEL_DRIFT = np.array([0.10, 0.06, -0.13, -0.03])   # 年末相对年初的占比变化

# 会员等级：占比、价格系数
MEMBER_LEVELS = ["普通会员", "银牌会员", "金牌会员", "钻石会员"]
MEMBER_PROB = np.array([0.55, 0.25, 0.15, 0.05])
MEMBER_PRICE_COEF = np.array([1.00, 1.05, 1.12, 1.20])

# 大促活动：名称、起止日期、折扣率下限、折扣率上限
PROMOTIONS = [
    ("年货节", "2025-01-10", "2025-01-26", 0.12, 0.22),
    ("38女神节", "2025-03-05", "2025-03-09", 0.10, 0.20),
    ("618大促", "2025-05-26", "2025-06-20", 0.15, 0.34),
    ("818促销", "2025-08-16", "2025-08-20", 0.08, 0.16),
    ("国庆黄金周", "2025-09-30", "2025-10-08", 0.10, 0.20),
    ("双11大促", "2025-10-20", "2025-11-13", 0.18, 0.40),
    ("双12促销", "2025-12-08", "2025-12-13", 0.12, 0.24),
]

# 每日下单时段权重（0—23 点），呈现上午、午后、晚间三个高峰
HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.9, 1.6, 2.4, 2.9, 2.6,
    2.2, 2.4, 2.6, 2.5, 2.4, 2.2, 2.6, 3.0, 3.2, 2.8, 1.8, 0.9,
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

# 数量分布（长尾）
QUANTITY_VALUES = np.array([1, 2, 3, 4, 5])
QUANTITY_PROB = np.array([0.62, 0.22, 0.10, 0.04, 0.02])

# 已知脏数据注入比例（写入 injection_log.json，供清洗环节对照）
INJECTION_RATE = {
    "duplicate_rows": 0.006,        # 重复订单
    "amount_outlier_high": 0.0025,  # 金额异常偏大
    "amount_outlier_low": 0.0015,   # 金额异常偏小
    "amount_string_format": 0.020,  # 金额被写成文本 "1234.00"
    "quantity_missing": 0.008,      # 数量缺失
    "quantity_negative": 0.004,     # 数量为负
    "rating_missing": 0.080,        # 评分缺失
    "region_missing": 0.020,        # 省份缺失
    "whitespace_case": 0.012,       # 渠道/品类含空格或大小写不一致
    "time_format_mixed": 0.003,     # 时间格式不统一
}

N_DAYS = 365
BASE_DAILY_ORDERS = 560
N_CUSTOMERS = 20000


# =====================================================================
# 二、时间维度：日订单量模型
# =====================================================================

def _gaussian_bump(doy, center, amplitude, width):
    """在一年中某一天附近形成钟形脉冲，用于模拟节假日效应。"""
    return amplitude * np.exp(-0.5 * ((doy - center) / width) ** 2)


def build_daily_volume(dates, rng):
    """按业务机理生成每天的订单量：
    订单量 = 基线 x 年度增长趋势 x 周周期 x 季节波动 x 节假日脉冲 x 对数正态扰动
    """
    doy = np.array([d.timetuple().tm_yday for d in dates])          # 一年中的第几天
    weekday = np.array([d.weekday() for d in dates])                # 0 = 周一
    trend = 1.0 + 0.38 * np.arange(N_DAYS) / (N_DAYS - 1)           # 全年增长约 38%

    weekly = np.where(weekday >= 5, 1.30, 1.0)                      # 周末高峰
    weekly = np.where(weekday == 4, 1.10, weekly)                   # 周五略高

    # 季节波动：夏季与年末两段旺季
    seasonal = (1.0
                + 0.16 * np.sin(2 * np.pi * (doy - 60) / 365.0)
                + 0.10 * np.sin(4 * np.pi * (doy - 60) / 365.0))

    # 节假日脉冲与春节低谷
    holiday = np.ones(N_DAYS)
    holiday += _gaussian_bump(doy, 1, 0.20, 3)      # 元旦
    holiday += _gaussian_bump(doy, 29, -0.42, 8)    # 春节前后物流停摆
    holiday += _gaussian_bump(doy, 67, 0.26, 4)     # 38 节
    holiday += _gaussian_bump(doy, 169, 0.72, 6)    # 618
    holiday += _gaussian_bump(doy, 230, 0.22, 3)    # 818
    holiday += _gaussian_bump(doy, 276, 0.30, 5)    # 国庆
    holiday += _gaussian_bump(doy, 315, 0.95, 5)    # 双 11
    holiday += _gaussian_bump(doy, 346, 0.34, 3)    # 双 12

    noise = rng.lognormal(mean=0.0, sigma=0.075, size=N_DAYS)
    volume = BASE_DAILY_ORDERS * trend * weekly * seasonal * holiday * noise
    return np.maximum(volume.round().astype(int), 80)


def mark_promotions(dates):
    """标记每个日期属于哪个大促活动，不在活动期则为空字符串。"""
    promo_name = np.array([""] * len(dates), dtype=object)
    for name, start, end, _d_low, _d_high in PROMOTIONS:
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        mask = np.array([(s <= d <= e) for d in dates])
        promo_name[mask] = name
    return promo_name


# =====================================================================
# 三、主体生成流程
# =====================================================================

def generate_orders(seed=RANDOM_SEED):
    """生成全年订单明细（干净版本），返回 DataFrame 与随机数发生器。"""
    rng = np.random.default_rng(seed)
    dates = list(pd.date_range("2025-01-01", "2025-12-31").date)
    daily_volume = build_daily_volume(dates, rng)
    total = int(daily_volume.sum())
    print(f"[1/5] 计划生成订单 {total:,} 条，日均 {daily_volume.mean():.0f} 条")

    # ---------- 把"天"展开成"一行一笔订单" ----------
    day_index = np.repeat(np.arange(N_DAYS), daily_volume)
    order_date = np.array(dates, dtype=object)[day_index]

    # ---------- 小时与分钟：按真实下单时段分布 ----------
    hour = rng.choice(24, size=total, p=HOUR_WEIGHTS)
    minute = rng.integers(0, 60, size=total)
    second = rng.integers(0, 60, size=total)
    order_time = pd.to_datetime({
        "year": [d.year for d in order_date],
        "month": [d.month for d in order_date],
        "day": [d.day for d in order_date],
        "hour": hour, "minute": minute, "second": second,
    })

    # ---------- 大促标记 ----------
    promo = mark_promotions(dates)[day_index]
    is_promo = promo != ""

    # ---------- 渠道：份额随时间漂移 ----------
    progress = day_index / (N_DAYS - 1)
    channel_prob = np.clip(CHANNEL_BASE[None, :] + CHANNEL_DRIFT[None, :] * progress[:, None], 0.01, None)
    channel_prob = channel_prob / channel_prob.sum(axis=1, keepdims=True)
    channel_daily = np.array([rng.choice(CHANNELS, p=channel_prob[i]) for i in range(N_DAYS)])
    channel = channel_daily[day_index]

    # ---------- 客户：先构造客户画像，再由客户产生订单 ----------
    cust_activity = rng.gamma(shape=1.6, scale=1.0, size=N_CUSTOMERS)   # 重尾：少数高活跃客户
    cust_member = rng.choice(len(MEMBER_LEVELS), size=N_CUSTOMERS, p=MEMBER_PROB)
    cust_region = rng.choice(len(REGIONS), size=N_CUSTOMERS, p=REGION_WEIGHTS)
    cust_city = np.array([rng.choice(REGIONS[c][1]) for c in cust_region], dtype=object)
    cust_weight = cust_activity * (1.0 + 0.25 * cust_member)            # 高等级会员更活跃
    cust_weight = cust_weight / cust_weight.sum()

    customer_idx = rng.choice(N_CUSTOMERS, size=total, p=cust_weight)
    customer_id = np.array([f"C{100000 + i}" for i in customer_idx], dtype=object)
    member_level = np.array(MEMBER_LEVELS, dtype=object)[cust_member[customer_idx]]
    member_coef = MEMBER_PRICE_COEF[cust_member[customer_idx]]
    province = np.array([REGIONS[c][0] for c in cust_region], dtype=object)[customer_idx]
    city = cust_city[customer_idx]
    region_coef = np.array([REGIONS[c][2] for c in cust_region])[customer_idx]

    # ---------- 品类与商品 ----------
    cat_names = list(CATEGORIES.keys())
    cat_prob = np.array([CATEGORIES[c]["weight"] for c in cat_names])
    cat_prob = cat_prob / cat_prob.sum()
    cat_idx = rng.choice(len(cat_names), size=total, p=cat_prob)
    category = np.array(cat_names, dtype=object)[cat_idx]
    product = np.array([rng.choice(CATEGORIES[cat_names[i]]["products"]) for i in cat_idx], dtype=object)

    # ---------- 单价：对数正态分布，受地区与会员等级调节 ----------
    price_mu = np.array([CATEGORIES[c]["price_mu"] for c in cat_names])[cat_idx]
    price_sigma = np.array([CATEGORIES[c]["price_sigma"] for c in cat_names])[cat_idx]
    unit_price = np.round(rng.lognormal(mean=price_mu, sigma=price_sigma) * region_coef * member_coef, 2)

    quantity = rng.choice(QUANTITY_VALUES, size=total, p=QUANTITY_PROB)

    # ---------- 折扣：大促期间明显加深 ----------
    promo_rate = {p[0]: p[3] for p in PROMOTIONS}      # 折扣率下限
    promo_rate_hi = {p[0]: p[4] for p in PROMOTIONS}   # 折扣率上限
    low = np.array([promo_rate.get(n, 0.0) for n in promo])
    high = np.array([promo_rate_hi.get(n, 0.0) for n in promo])
    promo_discount_rate = rng.uniform(low, high + 1e-9, size=total)
    base_discount_rate = rng.beta(2.0, 22.0, size=total) * 0.10        # 平日 0—10%
    discount_rate = np.where(is_promo, promo_discount_rate, base_discount_rate)

    gross = unit_price * quantity
    discount = np.round(gross * discount_rate, 2)
    amount = np.round(gross - discount, 2)

    # ---------- 满意度、评分与退货 ----------
    cat_rating = np.array([CATEGORIES[c]["rating_mu"] for c in cat_names])[cat_idx]
    satisfaction = np.clip(rng.normal(cat_rating, 0.78), 1.0, 5.0)
    rating = np.clip(np.round(satisfaction), 1, 5).astype(int)

    cat_return = np.array([CATEGORIES[c]["return_rate"] for c in cat_names])[cat_idx]
    return_prob = cat_return * (1.0 + 0.95 * (4.5 - satisfaction) / 2.0) * np.where(is_promo, 1.35, 1.0)
    return_prob *= np.where(quantity >= 3, 1.15, 1.0)                  # 多件订单退货概率略高
    is_return = rng.random(total) < np.clip(return_prob, 0.0, 0.6)

    df = pd.DataFrame({
        "order_id": [f"SO{d:%Y%m%d}{i:06d}" for d, i in zip(order_date, np.arange(total) % 1000000)],
        "order_time": order_time,
        "customer_id": customer_id,
        "province": province,
        "city": city,
        "channel": channel,
        "category": category,
        "product": product,
        "unit_price": unit_price,
        "quantity": quantity,
        "discount": discount,
        "amount": amount,
        "rating": rating,
        "is_return": is_return,
        "member_level": member_level,
        "promotion": promo,
    })
    print(f"[2/5] 基础数据生成完成，字段 {df.shape[1]} 个")
    return df, rng


# =====================================================================
# 四、注入已知脏数据（模拟真实业务系统的数据质量问题）
# =====================================================================

def inject_dirty_data(df, rng):
    """按预定比例注入 10 类数据质量问题，并返回注入清单。"""
    df = df.copy()
    n = len(df)
    log = {}

    def take(rate, exclude=None):
        """随机抽取若干行下标，并排除已经处理过的行，避免问题叠加。"""
        k = int(n * rate)
        idx = rng.choice(n, size=k, replace=False)
        if exclude is not None and len(exclude) > 0:
            idx = np.setdiff1d(idx, exclude)
        return idx

    # 1) 重复订单：复制整行（order_id 也随之重复）
    dup_idx = take(INJECTION_RATE["duplicate_rows"])
    log["duplicate_rows"] = len(dup_idx)

    # 2) 金额异常偏大 / 偏小
    hi_idx = take(INJECTION_RATE["amount_outlier_high"], exclude=dup_idx)
    lo_idx = take(INJECTION_RATE["amount_outlier_low"], exclude=np.concatenate([dup_idx, hi_idx]))
    amount = df["amount"].to_numpy(dtype=float)
    amount[hi_idx] *= rng.uniform(30, 60, size=len(hi_idx))
    amount[lo_idx] *= 0.01
    df["amount"] = np.round(amount, 2)
    log["amount_outlier_high"] = len(hi_idx)
    log["amount_outlier_low"] = len(lo_idx)

    # 3) 金额被写成带货币符号与千分位的文本
    str_idx = take(INJECTION_RATE["amount_string_format"])
    df["amount"] = df["amount"].astype(object)
    for i in str_idx:
        df.at[i, "amount"] = "¥{:,.2f}".format(float(df.at[i, "amount"]))
    log["amount_string_format"] = len(str_idx)

    # 4) 数量缺失 / 为负
    q_missing = take(INJECTION_RATE["quantity_missing"])
    q_neg = take(INJECTION_RATE["quantity_negative"], exclude=q_missing)
    df["quantity"] = df["quantity"].astype(float)
    df.loc[df.index[q_missing], "quantity"] = np.nan
    df.loc[df.index[q_neg], "quantity"] = -rng.integers(1, 4, size=len(q_neg))
    log["quantity_missing"] = len(q_missing)
    log["quantity_negative"] = len(q_neg)

    # 5) 评分缺失
    r_missing = take(INJECTION_RATE["rating_missing"])
    df["rating"] = df["rating"].astype(float)
    df.loc[df.index[r_missing], "rating"] = np.nan
    log["rating_missing"] = len(r_missing)

    # 6) 省份与城市缺失
    p_missing = take(INJECTION_RATE["region_missing"])
    df["province"] = df["province"].astype(object)
    df["city"] = df["city"].astype(object)
    df.loc[df.index[p_missing], "province"] = ""
    df.loc[df.index[p_missing], "city"] = ""
    log["region_missing"] = len(p_missing)

    # 7) 渠道 / 品类出现空格与大小写不一致
    ws_idx = take(INJECTION_RATE["whitespace_case"])
    half = len(ws_idx) // 2
    df["channel"] = df["channel"].astype(object)
    df["category"] = df["category"].astype(object)
    for i in ws_idx[:half]:
        df.at[i, "channel"] = " " + str(df.at[i, "channel"]).lower() + " "
    for i in ws_idx[half:]:
        df.at[i, "category"] = "  " + str(df.at[i, "category"]) + " "
    log["whitespace_case"] = len(ws_idx)

    # 8) 时间格式不统一（部分记录使用斜杠格式）
    t_idx = take(INJECTION_RATE["time_format_mixed"])
    df["order_time"] = df["order_time"].astype(object)
    for i in t_idx:
        df.at[i, "order_time"] = pd.Timestamp(df.at[i, "order_time"]).strftime("%Y/%m/%d %H:%M")
    log["time_format_mixed"] = len(t_idx)

    # 最后追加重复行并打乱顺序，避免重复行扎堆影响读者判断
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df, log


def main():
    df_clean, rng = generate_orders()
    df_raw, log = inject_dirty_data(df_clean, rng)
    print(f"[3/5] 脏数据注入完成，共 {sum(log.values()):,} 处问题")

    df_raw.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    print(f"[4/5] 原始数据已保存：{RAW_FILE}（{len(df_raw):,} 行 x {df_raw.shape[1]} 列）")

    log_payload = {
        "random_seed": RANDOM_SEED,
        "rows_generated": int(len(df_clean)),
        "rows_after_injection": int(len(df_raw)),
        "injection_rate": INJECTION_RATE,
        "injected_counts": {k: int(v) for k, v in log.items()},
        "promotions": [p[0] for p in PROMOTIONS],
        "note": "本清单记录人为注入的已知问题，用于清洗环节做前后对照验证。",
    }
    with open(RAW_DIR / "injection_log.json", "w", encoding="utf-8") as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    print(f"[5/5] 注入清单已保存：{RAW_DIR / 'injection_log.json'}")
    return df_raw


if __name__ == "__main__":
    main()
