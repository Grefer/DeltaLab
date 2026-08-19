# _*_ coding: utf-8 _*_
"""期权结构说明文档：结构页左侧那段纯文本。

按 (期权类型显示名, 子类型内部键) 索引，与 ``constants.OPTION_CLASSES`` 的
键一一对应。单独成文件是因为它有 200 行，混在常量表里会把真正需要经常改的
映射压到屏幕外。

**标题行不写在正文里**，由 ``SUBTYPE_DISPLAY`` 现生成——见 ``_title``。此前
每条正文自带一行 ``【中文名 内部键】``，于是同一个中文名在 ``constants`` 与
这里各有一份手抄副本，13 条累计结构漂开了 5 条（``ASGQ_call_put`` 显示名是
「到期熔断保障累计」而说明页写「到期观察熔断保障累计」是其中之一）。改成生成
之后这类漂移不可能再发生，``test_structure_docs`` 也把它锁住了。

因此本模块不再是零依赖——它 import ``constants``。方向是安全的：``constants``
只依赖 ``pricing``，从不反过来引用说明文档。
"""

from deltalab_ui.constants import SUBTYPE_DISPLAY


def _title(subtype):
    """``【敲出终止累计 Opt_Decumulator】``。

    中文名跟着 ``SUBTYPE_DISPLAY`` 走；内部键一律写全（此前 ASGQ 系列省掉了
    ``Opt_`` 前缀，而用户要拿它去对 ``Option_DE`` 的方法名和 GUI_USAGE 的表）。
    """
    return f"【{SUBTYPE_DISPLAY.get(subtype, subtype)} {subtype}】"


# 正文（不含标题行）。缩进对齐的那些冒号是有意的：说明页是等宽字体。
_DOC_BODIES = {
    ("香草期权 (Vanilla)", "Eu"): (
        "• Payoff: Call max(S_T−K,0) / Put max(K−S_T,0)\n"
        "• 定价: Black-Scholes 封闭解\n\n"
        "风险特征:\n"
        "  Delta 单调 0→1 (call) 或 −1→0 (put)\n"
        "  Gamma 集中于 ATM (S≈K), 随 T 缩小放大\n"
        "  Vega 对 ATM 最敏感, 随 √T 增长\n"
        "  Theta 为买方持续付出的时间价值"
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator"): (
        "每日观察 + 每日结算, 敲出即终止存续:\n"
        "  • 首次 S ≥ H (敲出): 当日及之后均停止累计\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "与「敲出计零累计」的差异: 那一个敲出仅当日计 0、后续仍继续\n"
        "观察; 本结构敲出即彻底了结, 路径依赖更强."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Back"): (
        "每日观察 + 每日结算, 三段式 cashflow:\n"
        "  • S ≥ H  (敲出障碍):  当日 0 赔付, 之后仍继续观察\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "总损益 = 所有观察日折现加总.\n"
        "卖方希望标的震荡于 (K, H) 区间, 触 K 承 N 倍下行."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix"): (
        "结构同「敲出计零累计」, 差异:\n"
        "  • K < S < H 区间按区间赔付 `fix` 结算, 而非 (S−K)\n"
        "  • 敲出段/杠杆段逻辑不变\n\n"
        "锁定中间段现金流, 便于账务管理."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix_E"): (
        "结构同「敲出计零·区间固赔累计」, 差异只在杠杆腿:\n"
        "  • K ≤ S < H 区间: 每日区间赔付 `fix`\n"
        "  • 杠杆腿 (S ≤ K): 不每日结算, 仅按到期日收盘价\n"
        "    一次性结算 (S_T − K) × 累计天数 × N\n\n"
        "适合到期一次性交割杠杆腿的场景."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator"): (
        "三段式每日结算:\n"
        "  • S ≥ H :  (S − H) 1 倍  (敲出后仍给买方正向收益)\n"
        "  • K < S < H: (S − K) 1 倍\n"
        "  • S ≤ K :  (S − K) N 倍\n\n"
        "相比「敲出计零累计」, 保留敲出后的上行收益, 故称'增强'."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator_Fix"): (
        "  • S ≥ H :  (S − H) 1 倍\n"
        "  • K < S < H: 区间赔付 `fix`\n"
        "  • S ≤ K :  (S − K) N 倍"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_call_put"): (
        "路径依赖 + 到期一次性结算:\n"
        "  • 若路径曾 S ≥ H (熔断): 熔断日后统一按 (S_T − P)\n"
        "  • 从未熔断: 按 (S_T − K), 若 S_T ≤ K 额外 N 倍\n\n"
        "保障价 P 提供下行软保护.\n"
        "13 个累计结构里唯一主项不逐日结算的一个——其余熔断族\n"
        "都是每日结算, 只有杠杆腿的观察频率有别."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DP"): (
        "每日观察 + 每日结算, 熔断后改按保障价 P:\n"
        "  • 未熔断  : 每日 (S − K), S ≤ K 的当日乘 N 倍\n"
        "  • 熔断日起: 每日 (S − P) 1 倍\n\n"
        "与「熔断保障·到期杠杆累计」的差异**只在杠杆腿**:\n"
        "这一个逐日看 S ≤ K, 因此 Delta/Gamma 沿路径跳得更碎;\n"
        "熔断判定两者完全一样, 都是逐日路径依赖."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EP"): (
        "每日结算, 熔断后改按保障价 P:\n"
        "  • 未熔断  : 每日 (S − K) 1 倍\n"
        "  • 熔断日起: 每日 (S − P)\n"
        "  • 杠杆腿  : 只在到期日观察——未熔断且 S_T ≤ K 时,\n"
        "    一次性追加 (S_T − K) × 累计天数 × (N − 1)\n\n"
        "与「熔断保障·每日杠杆累计」的差异**只在杠杆腿**;\n"
        "熔断判定两者完全一样, 都是逐日路径依赖."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DF"): (
        "每日观察 + 每日结算, 熔断后改按熔断赔付结算:\n"
        "  • 未熔断  : 每日 (S − K), S ≤ K 的当日乘 N 倍\n"
        "  • 熔断日起: 每日熔断赔付 `amount`"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EF"): (
        "每日结算, 熔断后改按熔断赔付结算:\n"
        "  • 未熔断  : 每日 (S − K) 1 倍\n"
        "  • 熔断日起: 每日熔断赔付 `amount`\n"
        "  • 杠杆腿  : 只在到期日观察——未熔断且 S_T ≤ K 时,\n"
        "    一次性追加 (S_T − K) × 累计天数 × (N − 1)"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DFF"): (
        "每日观察 + 每日结算, 区间赔付 + 熔断赔付:\n"
        "  • K < S < H (区间): 每日区间赔付 `fix`\n"
        "  • S ≤ K           : 每日 (S − K) N 倍\n"
        "  • 熔断日起        : 每日熔断赔付 `amount`\n\n"
        "`fix` 与 `amount` 是两笔不同的钱: 前者按标的落在区间内\n"
        "触发, 后者按熔断触发, 熔断之后不再有区间赔付."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EFF"): (
        "每日结算, 区间赔付 + 熔断赔付:\n"
        "  • K < S < H (区间): 每日区间赔付 `fix`\n"
        "  • S ≤ K           : 每日 (S − K) 1 倍\n"
        "  • 熔断日起        : 每日熔断赔付 `amount`\n"
        "  • 杠杆腿          : 只在到期日观察——未熔断且 S_T ≤ K 时,\n"
        "    一次性追加 (S_T − K) × 累计天数 × (N − 1)"
    ),
    ("亚式期权 (Asian)", "Asian"): (
        "Payoff = clip( mean(S[-N:]) − K,  minPay,  maxPay ) × cp\n\n"
        "  • 取最后 N 个交易日均价与 K 的差额\n"
        "  • minPay / maxPay 限定赔付区间\n"
        "  • 平均化显著降低末日价格风险\n"
        "  • Gamma / Vega 远低于同期限 Vanilla"
    ),
    ("亚式期权 (Asian)", "EnhanceAsian"): (
        "每日先做价格增强:\n"
        "  • Call: 观察价 = max(S, E)\n"
        "  • Put : 观察价 = min(S, E)\n"
        "再求均值与 K 比较, 并 clip 到 [minPay, maxPay].\n\n"
        "E 提供'每日保底'效果, 提升买方期望."
    ),
    ("气囊期权 (Airbag)", "Opt_Airbag"): (
        "到期结算, 路径判断是否敲入 KI:\n"
        "  • 未敲入 (Call: min(S) > KI):  pr × max(S_T − K, 0)\n"
        "  • 已敲入                    : pr_ki × (S_T − K)\n\n"
        "小幅下行时买方有软垫保护 (payoff=0 而非负);\n"
        "一旦跌破 KI, 转为线性承担下行, 即'气囊爆掉'."
    ),
    ("雪球期权 (Snowball)", "Opt_Snowball"): (
        "(cp=-1 雪球 / cp=1 反雪球)\n"
        "MC 定价, 路径依赖, 四种到期情形:\n"
        "  • 未敲入未敲出: 全期票息 s00 × coupon × 期限\n"
        "  • 敲出 (观察日触 KO): 敲出票息 × 持有期, 提前了结\n"
        "  • 敲入且敲出: 同敲出 (敲出优先)\n"
        "  • 敲入未敲出: 追保=承担完整亏损; 不追保=亏损按 margin × s00 封顶\n\n"
        "敲入逐日监测; 敲出按固定间隔观察 (首次=锁定期后, 末次=到期日).\n"
        "每期降敲(ko_step>0)时 KO 自期初值逐期递减, 越往后越易敲出.\n"
        "卖方 (持有者) 短 vega/gamma、正 theta; 现价 ↗ 近 KO 时\n"
        "Delta 与 Gamma 易出现剧烈跳变 (敲出悬崖)."
    ),
}

STRUCTURE_DOCS = {
    (cls_name, subtype): f"{_title(subtype)}\n{body}"
    for (cls_name, subtype), body in _DOC_BODIES.items()
}
