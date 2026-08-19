# _*_ coding: utf-8 _*_
"""界面层的静态注册表与显示名映射。

三类内容：期权类型注册表（``OPTION_CLASSES``，含各结构的构造闭包）、GUI 显示
名 ↔ 后端内部键的双向映射、以及图表配色/上限这类展示常量。

这里只依赖 ``pricing``，不依赖界面层其他模块，也不 import ``gui_app``——
``OPTION_CLASSES`` 几乎被界面每个角落引用，它一旦反向依赖就必然成环。
"""

import numpy as np
import history_store
from pricing import Option_AB, Option_AS, Option_DE, Option_SNB, Option_Vanilla


# ============================================================
#  期权类型注册表
# ============================================================

def _snowball_ko_observ(T, first_obs, period):
    """按"锁定期 + 固定间隔 + 末次=到期"生成敲出观察交易日序号（1-based）。

    全用交易日：首个观察在第 first_obs 日，其后每 period 个交易日一次，并
    强制最后一次落在到期日 T（与到期不齐时末段为短桩）。返回升序去重列表。
    """
    T = int(T)
    first_obs = max(1, int(first_obs))
    period = max(1, int(period))
    days = list(range(first_obs, T + 1, period))
    if not days or days[-1] != T:
        days.append(T)                         # 末次观察 = 到期
    return sorted({d for d in days if 1 <= d <= T})


def _build_snowball(st, p):
    """构造雪球（act=1 交易日计息）。观察日按锁定期+固定间隔生成（按完整交易日计算）；
    ko_step>0 时为降敲，KO 自期初值每观察日递减 ko_step 个点，得逐观察日 KO 向量。"""
    T = int(p["T"])
    ko_observ = _snowball_ko_observ(T, p["first_obs_d"], p["obs_period_d"])
    step = float(p.get("ko_step", 0.0) or 0.0)
    KO = [p["KO"] - step * i for i in range(len(ko_observ))] if step else p["KO"]
    return Option_SNB(
        st, p["s00"], p["s0"], p["K"], p["KI"], KO, T,
        p["sigma"], p["coupon"], p["coupon_ko"], p["margin"], 1, p["cp"],
        r=p["r"], q=p["q"], sr=[], ko_observ=ko_observ, nPath=p["nPath"],
        margin_call=bool(p["margin_call"]),
    )


def _parse_number_sequence(raw, label):
    """把「1, 2.5, 3」这样的输入解析成 float 列表；留空即空列表。

    逗号、空格、中文逗号都当分隔符——用户从行情软件复制出来的价格串
    什么分隔符都可能有，为此报错没有意义。
    """
    if raw is None:
        return []
    # 已经是序列就逐项校验，不要先 str() 再拆——那会把 "[100.0, 101.0]"
    # 拆成 "[100.0" 这种碎片。快照重放与测试都是直接传列表进来的。
    if isinstance(raw, (list, tuple, np.ndarray)):
        tokens = [str(item) for item in raw]
        return _validated_numbers(tokens, label)
    text = str(raw).replace("，", ",").replace(";", ",")
    tokens = [t for t in text.replace(",", " ").split() if t]
    return _validated_numbers(tokens, label)


def _validated_numbers(tokens, label):
    values = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"{label} 含无法解析为数值的项: {token!r}") from None
        if not np.isfinite(value):
            raise ValueError(f"{label} 含非有限数值: {token!r}")
        if value <= 0:
            raise ValueError(f"{label} 的价格必须为正: {token!r}")
        values.append(value)
    return values


# 方向敏感的价格档位。障碍类结构的触发条件会随 cp 翻转方向——例如累计期权
# 看涨时 `S >= H` 熔断、看跌时 `S <= H` 熔断——所以同一个 H 在两个方向上要摆在
# 标的价的两侧才有意义。默认值只朝一个方向摆着，切到另一方向就会「首日必触发」，
# 算出来是一串退化值，界面上却看不出异常。
#
# 每一项给出：(需要绕 s0 镜像的档位字段, 用于判定是否已被触发的规则)。
# 规则是 (字段, 看涨时该在 s0 的哪一侧) —— "above" 表示看涨时必须高于 s0。
DIRECTIONAL_LEVELS = {
    "累计期权 (Decumulator)": {
        "mirror": ("K", "H"),
        "barriers": (("H", "above", "熔断价格"),),
    },
    "气囊期权 (Airbag)": {
        "mirror": ("K", "KI"),
        "barriers": (("KI", "below", "敲入价格"),),
    },
    "雪球期权 (Snowball)": {
        # 雪球默认 cp=-1，方向语义与另外两类相反：cp=-1 时敲入在下、敲出在上。
        "mirror": ("K", "KI", "KO"),
        "barriers": (("KI", "below", "敲入价格"), ("KO", "above", "敲出价格")),
        "call_is_reversed": True,
    },
}


OPTION_CLASSES = {
    "香草期权 (Vanilla)": {
        "class": Option_Vanilla,
        "subtypes": ["Eu"],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 100.0),
            ("T_days", "期限(交易日)",   int,   22),
            ("sigma",  "波动率",        float, 0.18),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
        ],
        "build": lambda st, p: Option_Vanilla(
            st, p["s0"], [], p["K"], p["T_days"],
            p["sigma"], p["cp"],
            r=p["r"], q=p["q"], exe_mode=st,
        ),
    },
    "累计期权 (Decumulator)": {
        "class": Option_DE,
        # 顺序即下拉顺序：先按「触碰障碍之后怎么办」分成敲出族 / 熔断族，族
        # 内让 D/E（杠杆腿每日观察 vs 到期观察）两两相邻——这一对正是最容易
        # 选错的地方，隔开摆就得在下拉里来回翻着比。
        "subtypes": [
            "Opt_Decumulator", "Opt_Decumulator_Back",
            "Opt_Decumulator_Fix", "Opt_Decumulator_Fix_E",
            "Opt_EnDecumulator", "Opt_EnDecumulator_Fix",
            "Opt_ASGQ_call_put",
            "Opt_ASGQ_DP", "Opt_ASGQ_EP",
            "Opt_ASGQ_DF", "Opt_ASGQ_EF",
            "Opt_ASGQ_DFF", "Opt_ASGQ_EFF",
        ],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 90.0),
            ("T_days", "剩余期限(交易日)", int, 20),
            ("T_over", "已过天数",       int,   0),
            # 与「已过天数」配对：这几天的收盘价。两者必须等长——已实现的天数
            # 决定了模拟段从哪一天接上。默认都是 0 / 空，此时行为与从前完全一致。
            ("sr",     "已实现序列(逗号分隔)", list,  ""),
            ("sigma",  "波动率",        float, 0.18),
            ("H",      "障碍价格",       float, 110.0),
            ("N",      "杠杆倍数",       int,   2),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("fix",    "区间赔付(部分结构必填)", float, 0.0),
            ("P",      "保障价格(部分结构必填)", float, 100.0),
            ("amount", "熔断赔付(部分结构必填)", float, 0.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_DE(
            st, p["s0"], _parse_number_sequence(p["sr"], "已实现序列"),
            p["K"], p["T_over"], p["T_days"],
            list(range(1, p["T_days"] + p["T_over"] + 1)),
            p["sigma"], p["H"], p["N"], p["cp"],
            r=p["r"], q=p["q"], nPath=p["nPath"],
            fix=p["fix"] if p["fix"] else None,
            P=p["P"] if p["P"] else None,
            amount=p["amount"] if p["amount"] else None,
        ),
    },
    "亚式期权 (Asian)": {
        "class": Option_AS,
        "subtypes": ["Asian", "EnhanceAsian"],
        "params": [
            ("s0",     "初始价格 S0",    float, 100.0),
            ("K",      "行权价",        float, 100.0),
            ("E",      "增强价(Enhanced)", float, 100.0),
            ("T",      "期限(交易日)",   int,   22),
            ("N",      "观察日数",       int,   22),
            ("sigma",  "波动率",        float, 0.15),
            ("cp",     "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("minPay", "最低赔付",       float, 0.0),
            ("maxPay", "最高赔付",       float, 999999.0),
            ("r",      "无风险利率",     float, 0.03),
            ("q",      "分红率",        float, 0.03),
            ("nPath",  "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_AS(
            st, p["s0"], [], p["K"], p["E"], p["T"], p["N"],
            p["sigma"], p["cp"], p["minPay"], p["maxPay"],
            r=p["r"], q=p["q"], nPath=p["nPath"]
        ),
    },
    "气囊期权 (Airbag)": {
        "class": Option_AB,
        "subtypes": ["Opt_Airbag"],
        "params": [
            ("s0",    "初始价格 S0",    float, 100.0),
            ("K",     "行权价",        float, 100.0),
            ("KI",    "敲入价",        float, 90.0),
            ("T_days","期限(交易日)",   int,   20),
            ("sigma", "波动率",        float, 0.18),
            ("pr",    "参与率",        float, 0.8),
            ("pr_ki", "敲入参与率",     float, 1.0),
            ("cp",    "方向",          int,   1, {"看涨 (Call)": 1, "看跌 (Put)": -1}),
            ("r",     "无风险利率",     float, 0.03),
            ("q",     "分红率",        float, 0.03),
            ("nPath", "定价路径数 (MC)", int,   100000),
        ],
        "build": lambda st, p: Option_AB(
            st, p["s0"], [], p["K"], p["KI"], p["T_days"],
            list(range(1, p["T_days"] + 1)),
            p["sigma"], p["pr"], p["pr_ki"], p["cp"],
            r=p["r"], q=p["q"], nPath=p["nPath"]
        ),
    },
    "雪球期权 (Snowball)": {
        "class": Option_SNB,
        "subtypes": ["Opt_Snowball"],
        "params": [
            ("s00",        "入场价 S00",        float, 100.0),
            ("s0",         "最新价 S0",         float, 100.0),
            ("K",          "行权价",            float, 100.0),
            ("KI",         "敲入价",            float, 80.0),
            ("KO",         "期初敲出价",         float, 103.0),
            ("T",          "剩余期限(交易日)",    int,   243),
            # 锁定期/观察间隔：值用交易日(与引擎一致)，下拉给月度预设辅助输入，
            # 可编辑——既能选预设也能手填自定义交易日数（21 交易日 ≈ 1 个月）。
            ("first_obs_d","首次敲出观察",        int,   63,
             {"锁1月 (21)": 21, "锁2月 (42)": 42, "锁3月 (63)": 63, "锁6月 (126)": 126},
             {"editable": True}),
            ("obs_period_d","观察间隔",          int,   21,
             {"月度 (21)": 21, "双月 (42)": 42, "季度 (63)": 63, "半年 (126)": 126},
             {"editable": True}),
            ("ko_step",    "每期降敲(点,0=平敲)", float, 0.0),
            ("sigma",      "波动率",            float, 0.15),
            ("coupon",     "未敲出票息率(年化)",  float, 0.15),
            ("coupon_ko",  "敲出票息率(年化)",    float, 0.15),
            ("margin_call", "保证金模式",          int,   1, {"追保(亏损不封顶)": 1, "不追保(有限亏损)": 0}),
            ("margin",     "保证金比例(不追保封顶)", float, 0.2),
            ("cp",         "方向",             int,   -1, {"雪球 (卖看跌)": -1, "反雪球 (卖看涨)": 1}),
            ("r",          "无风险利率",        float, 0.03),
            ("q",          "分红率",            float, 0.03),
            ("nPath",      "定价路径数 (MC)",    int,   20000),
        ],
        # act 固定为 1（交易日计息，无需交易日历）；观察日=锁定期+固定间隔+末次到期，
        # ko_step>0 时为降敲（逐观察日 KO 递减），见 _build_snowball。
        "build": _build_snowball,
    },
}


# ============================================================
#  GUI 显示名 ↔ 后端内部键 映射
#  说明：后端 (hedge_backtest / Option_* 类) 使用英文/方法名做字符串匹配，
#  这里仅影响界面显示；读取 Combobox 值后需通过 *_FROM_DISPLAY 反向映射
#  还原为内部键再传给后端。
# ============================================================

# 累计期权 13 个结构的中文名按三个**正交维度**拼，顺序固定，取默认值的那一
# 段省略不写：
#
#   ① 触碰障碍 H 之后怎么办（必写）
#        敲出终止 / 敲出计零 / 敲出增强 / 熔断保障 / 熔断赔付
#      「敲出」与「熔断」的分工是有意的：敲出族触碰后只影响赔付本身（永久
#      归零、当日归零、或仍付 S−H），熔断族触碰后**换一整套结算规则**跑到
#      到期（改按保障价 P，或改付熔断赔付 amount）。
#   ② 区间 (K, H) 怎么赔：线性 (S−K) 省略不写；按 `fix` 结算写「区间固赔」
#   ③ 杠杆腿 (S ≤ K) 怎么观察：「每日杠杆」逐日乘 N；「到期杠杆」只看到期
#      收盘价，一次性按累计天数结算
#
# 这一版之前，③ 这个维度的中文名全是错的：`ASGQ_E*` 与 `ASGQ_D*` 被写成
# 「到期观察熔断」与「每日观察熔断」，但七个 ASGQ 结构的熔断判定是同一行
# ``np.cumsum(ss >= H, axis=1) > 0``，逐日路径依赖、没有任何差别——E/D 的
# 真实差别只在杠杆腿。于是「熔断每日保障累计」(EP) 与「每日熔断保障累计」
# (DP) 不但只差两字语序、在下拉里认不出，两个还都指错了对象。
#
# 另外 fix 与 amount 曾共用「固赔」二字（`Decumulator_Fix` 的固赔是 fix、
# `ASGQ_EF` 的固赔是 amount、`*FF` 的「双固赔」是两个都有），同一个下拉里
# 三种含义。现在「区间固赔」只指 fix，amount 由 ① 的「熔断赔付」承担。
SUBTYPE_DISPLAY = {
    "Eu":                    "欧式",
    "Opt_Decumulator":       "敲出终止累计",
    "Opt_Decumulator_Back":  "敲出计零累计",
    "Opt_Decumulator_Fix":   "敲出计零·区间固赔累计",
    "Opt_Decumulator_Fix_E": "敲出计零·区间固赔·到期杠杆累计",
    "Opt_EnDecumulator":     "敲出增强累计",
    "Opt_EnDecumulator_Fix": "敲出增强·区间固赔累计",
    # 到期结算：主项也只用到期收盘价，是 13 个里唯一不逐日结算的。
    "Opt_ASGQ_call_put":     "熔断保障·到期结算累计",
    "Opt_ASGQ_DP":           "熔断保障·每日杠杆累计",
    "Opt_ASGQ_EP":           "熔断保障·到期杠杆累计",
    "Opt_ASGQ_DF":           "熔断赔付·每日杠杆累计",
    "Opt_ASGQ_EF":           "熔断赔付·到期杠杆累计",
    "Opt_ASGQ_DFF":          "熔断赔付·区间固赔·每日杠杆累计",
    "Opt_ASGQ_EFF":          "熔断赔付·区间固赔·到期杠杆累计",
    "Asian":                 "标准亚式",
    "EnhanceAsian":          "增强亚式",
    "Opt_Airbag":            "气囊",
    "Opt_Snowball":          "雪球",
}

# 改名前的旧显示名。只并进反向映射，不参与正向显示——界面各处一律显示新名，
# 但用旧名（手输、脚本、外部表格粘贴）仍能还原成内部键。落盘的结果里存的是
# 内部键，所以这张表不参与任何迁移，纯粹兜住输入侧。
_LEGACY_SUBTYPE_DISPLAY = {
    "Opt_Decumulator":       "普通累计",
    "Opt_Decumulator_Back":  "回归累计",
    "Opt_Decumulator_Fix":   "固定赔付回归累计",
    "Opt_Decumulator_Fix_E": "固赔到期结算累计",
    "Opt_EnDecumulator":     "增强回归累计",
    "Opt_EnDecumulator_Fix": "固定赔付增强累计",
    "Opt_ASGQ_call_put":     "到期熔断保障累计",
    "Opt_ASGQ_EP":           "熔断每日保障累计",
    "Opt_ASGQ_EF":           "熔断每日固赔累计",
    "Opt_ASGQ_EFF":          "熔断每日双固赔累计",
    "Opt_ASGQ_DP":           "每日熔断保障累计",
    "Opt_ASGQ_DF":           "每日熔断固赔累计",
    "Opt_ASGQ_DFF":          "每日熔断双固赔累计",
}

# 旧名先铺底，新名后覆盖：万一某个旧名与新名撞车，赢的必须是新名。
SUBTYPE_FROM_DISPLAY = {v: k for k, v in _LEGACY_SUBTYPE_DISPLAY.items()}
SUBTYPE_FROM_DISPLAY.update({v: k for k, v in SUBTYPE_DISPLAY.items()})

# 已保存结果列表要显示子类型的中文名。history_store 保持纯逻辑、不反
# 向依赖 GUI，所以由这里把映射注入进去。
history_store.SUBTYPE_DISPLAY = SUBTYPE_DISPLAY

STRATEGY_DISPLAY = {
    "close_to_close": "每日收盘",
    "fixed_times": "每日固定时刻",
    "hedge_band": "价格波动触发调仓",
}
STRATEGY_FROM_DISPLAY = {v: k for k, v in STRATEGY_DISPLAY.items()}


# Wind 仍需要明确的起止边界和 BarSize；GUI 用“自动（推荐）”把这组底层
# 参数按单次回测 / 历史择优语义解析后，再传给既有后端 API。
WIND_AUTO_BAR_SIZE = "自动（推荐）"
WIND_BAR_SIZE_OPTIONS = (
    WIND_AUTO_BAR_SIZE, "日频", "60分钟", "30分钟", "15分钟", "5分钟", "1分钟",
)
_WIND_BAR_MINUTES = {
    "60分钟": 60, "30分钟": 30, "15分钟": 15, "5分钟": 5, "1分钟": 1,
}
# 采样粒度只有 WIND_BAR_SIZE_OPTIONS 一套标签；自动推荐与 Wind 请求参数
# 都从 _WIND_BAR_MINUTES 派生，避免下拉改名后留下对不上的字面量。
_WIND_FIXED_TIME_BAR_LABELS = ("15分钟", "5分钟", "1分钟")
# 价格波动触发只比较 bar 收盘价（HedgeBandStrategy.should_hedge），bar 内
# 穿带后回落的行情完全不可见，且漏掉的方向是单边的：少调仓 -> 低估交易
# 成本 -> 高估策略表现。因此自动粒度一律取最细的 1 分钟，不按带宽分档：
# 分档会让同一个候选的评分依赖“本批次里最窄的候选是谁”，而宽带下省下
# 的数据量也换不到可测量的精度。带宽与粒度的关系只用于量化手动选粗时
# 的代价，见 _WIND_BAND_MISS_RATES。
_WIND_BAND_BAR_LABEL = "1分钟"
_WIND_DATE_BUFFER_DAYS = 21

# 排名依据的中英映射。两个口径都相对“日内不动”的基线取增量，区别只在于
# 看绝对多赚了多少，还是看这份增量的信噪比（每单位波动换来多少增量）。
HISTORY_OBJECTIVE_DISPLAY = {
    "incremental_pnl": "增量收益（赚更多）",
    "incremental_sharpe": "增量信噪比（更稳）",
}
HISTORY_OBJECTIVE_FROM_DISPLAY = {
    value: key for key, value in HISTORY_OBJECTIVE_DISPLAY.items()
}
# 表格里对应两个排名口径的列，其表头可点击切换排名依据。
_OBJECTIVE_COLUMN_KEYS = {
    "incremental_pnl": "incremental_pnl",
    "incremental_sharpe": "incremental_sharpe",
}
# ranking 里承载这两个口径的列名。品种池模式不产出它们（跨合约不能直接
# 把金额 PnL 相加），展示层据此降级——见 _build_history_metric_tree。
_OBJECTIVE_RANKING_COLUMNS = (
    "incremental_pnl_vs_c2c",
    "incremental_sharpe_vs_c2c",
)

# 图表口径固定为整段接续（模式 "full"），不再有模式下拉，因此这里只留
# 指标的显示名。history_selection 的模型层仍实现着 single / typical，供
# 直接调用，但界面不再提供入口——它们与排名口径不一致。
HISTORY_CHART_METRIC_DISPLAY = {
    "net": "净损益",
    "gross": "成本前损益",
    "tc": "交易成本",
}
HISTORY_CHART_METRIC_FROM_DISPLAY = {
    value: key for key, value in HISTORY_CHART_METRIC_DISPLAY.items()
}

# 结果对比页与策略优选页此前各有一套曲线配色，同一个策略在两页会拿到不同
# 颜色，用户无法把两页的曲线对应起来。两页现在共用这一份色表和标记表，由
# BacktestApp._strategy_style 按会话内首次出现顺序登记。
STRATEGY_CHART_COLORS = (
    "#2563EB", "#D97706", "#7C3AED", "#0F766E",
    "#DB2777", "#DC2626", "#0891B2", "#65A30D",
    "#9333EA", "#C2410C", "#4F46E5", "#047857",
)
STRATEGY_CHART_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
# 同一策略的多条结果共用**色相**（跨页同色是有意的），组内靠明度加线型分开。
# 换区间、换行情、换方向重跑同一个策略恰恰是结果对比页的头号用途，不分开的话
# 一勾多条就是一团完全同色的线。
STRATEGY_CHART_DASHES = (
    "solid", (0, (5, 2)), (0, (1, 1.4)), (0, (7, 2, 1.5, 2)))
# 组内明度档：原色 / 提亮 / 压暗。正数往白里混，负数往黑里压。
#
# **只要三档**是因为线型有四种，3 与 4 互质，(明度, 线型) 的组合到第 12 条才
# 重复，而图上最多同时画 8 条（MAX_COMPARISON_CHART_CURVES）——够用了。档数少
# 才调得开：实测六种基色下三档的最小亮度差是 34，摊成八档只剩 8，那种深浅在
# 691×226 px 的画布上根本读不出来。相邻两条因此明度与线型必定同时不同。
#
# 第 0 档恒为原色：跨页对照靠同一策略的第一条与策略优选页严格同色。
STRATEGY_CHART_SHADES = (0.0, 0.34, -0.40)
# 对比图同时画的曲线上限。指标表不受限——20 行照排照导，「全选 → 点列头
# 排序 → 看第一行」这条路要留着；闸门只加在图上：691×226 px 的画布、12 色取
# 模，20 条叠上去谁也认不出谁。数字与策略优选页的
# MAX_HISTORY_CHART_CANDIDATES 取齐，两页对"几条还看得清"该给同一个答案。
MAX_COMPARISON_CHART_CURVES = 8
# 每日收盘在两页都是固定基准，必须始终占用同一个颜色位，不能因为登记顺序
# 不同而换色。
BASELINE_STRATEGY_STYLE_KEY = "close_to_close"

# 快照来源：只有 origin 是结构化字段，可供结果池分组与回跳使用；此前来源
# 只体现在用户可改的结果名前缀里，改名即丢失。
SNAPSHOT_ORIGIN_MANUAL = "manual"
SNAPSHOT_ORIGIN_HISTORY_REPLAY = "history_replay"
SNAPSHOT_ORIGIN_DISPLAY = {
    SNAPSHOT_ORIGIN_MANUAL: "手工回测",
    SNAPSHOT_ORIGIN_HISTORY_REPLAY: "分段重放",
}

SIGMA_SOURCE_DISPLAY = {
    "implied":  "隐含波动率",
    "realized": "已实现波动率",
}
SIGMA_SOURCE_FROM_DISPLAY = {v: k for k, v in SIGMA_SOURCE_DISPLAY.items()}
