# _*_ coding: utf-8 _*_
"""期权结构说明文档：结构页左侧那段纯文本。

按 (期权类型显示名, 子类型内部键) 索引，与 ``constants.OPTION_CLASSES`` 的
键一一对应。纯数据、零依赖，单独成文件是因为它有 140 行，混在常量表里会把
真正需要经常改的映射压到屏幕外。
"""

STRUCTURE_DOCS = {
    ("香草期权 (Vanilla)", "Eu"): (
        "【欧式香草期权】\n"
        "• Payoff: Call max(S_T−K,0) / Put max(K−S_T,0)\n"
        "• 定价: Black-Scholes 封闭解\n\n"
        "风险特征:\n"
        "  Delta 单调 0→1 (call) 或 −1→0 (put)\n"
        "  Gamma 集中于 ATM (S≈K), 随 T 缩小放大\n"
        "  Vega 对 ATM 最敏感, 随 √T 增长\n"
        "  Theta 为买方持续付出的时间价值"
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator"): (
        "【普通累计 Opt_Decumulator】\n"
        "每日观察 + 每日结算, 敲出即终止存续:\n"
        "  • 首次 S ≥ H (敲出): 当日及之后均停止累计\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "与 Back 的差异: Back 敲出仅当日计 0、后续仍继续观察;\n"
        "本结构敲出即彻底了结, 路径依赖更强."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Back"): (
        "【回归累计 Opt_Decumulator_Back】\n"
        "每日观察 + 每日结算, 三段式 cashflow:\n"
        "  • S ≥ H  (敲出障碍):  当日 0 赔付\n"
        "  • K < S < H       :  1 倍 (S − K) 结算\n"
        "  • S ≤ K           :  N 倍杠杆 (S − K) 结算\n\n"
        "总损益 = 所有观察日折现加总.\n"
        "卖方希望标的震荡于 (K, H) 区间, 触 K 承 N 倍下行."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix"): (
        "【固定赔付回归累计 Opt_Decumulator_Fix】\n"
        "结构同 Back, 差异:\n"
        "  • K < S < H 区间按固定金额 `fix` 结算, 而非 (S−K)\n"
        "  • 敲出段/杠杆段逻辑不变\n\n"
        "锁定中间段现金流, 便于账务管理."
    ),
    ("累计期权 (Decumulator)", "Opt_Decumulator_Fix_E"): (
        "【固赔到期结算累计 Opt_Decumulator_Fix_E】\n"
        "结构同 Fix, 差异在杠杆段结算方式:\n"
        "  • K ≤ S < H 区间: 每日固定金额 `fix`\n"
        "  • 杠杆段 (S ≤ K): 不每日结算, 仅按到期日收盘价\n"
        "    一次性结算 (S_T − K) × 累计天数 × N\n\n"
        "适合到期一次性交割杠杆腿的固赔回归累计."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator"): (
        "【增强回归累计 Opt_EnDecumulator】\n"
        "三段式每日结算:\n"
        "  • S ≥ H :  (S − H) 1 倍  (敲出后仍给买方正向收益)\n"
        "  • K < S < H: (S − K) 1 倍\n"
        "  • S ≤ K :  (S − K) N 倍\n\n"
        "相比 Back, 保留敲出后上行收益, 故称'增强'."
    ),
    ("累计期权 (Decumulator)", "Opt_EnDecumulator_Fix"): (
        "【固定赔付增强累计 Opt_EnDecumulator_Fix】\n"
        "  • S ≥ H :  (S − H) 1 倍\n"
        "  • K < S < H: 固定金额 `fix`\n"
        "  • S ≤ K :  (S − K) N 倍"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_call_put"): (
        "【到期观察熔断保障累计 ASGQ_call_put】\n"
        "路径依赖 + 到期一次性结算:\n"
        "  • 若路径曾 S ≥ H (熔断): 熔断日后统一按 (S_T − P)\n"
        "  • 从未熔断: 按 (S_T − K), 若 S_T ≤ K 额外 N 倍\n\n"
        "保障价 P 提供下行软保护, ASGQ = 熔断保障累计."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EP"): (
        "【熔断保障累计(每日结算) ASGQ_EP】\n"
        "  • 未熔断部分: 按日 (S − K) 累加\n"
        "  • 熔断日起  : 每日 (S − P) 结算"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EF"): (
        "【熔断固定赔付累计 ASGQ_EF】\n"
        "  • 未熔断部分: 按日 (S − K) 累加\n"
        "  • 熔断日起  : 每日固定金额 `amount`"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_EFF"): (
        "【熔断每日双固赔累计 ASGQ_EFF】\n"
        "到期观察 + 每日结算, 双固定赔付:\n"
        "  • K < S < H (区间): 每日固定金额 `fix`\n"
        "  • S ≤ K           : 每日 (S − K) 1 倍\n"
        "  • 熔断日起        : 每日固定金额 `amount`\n"
        "  • 到期 S_T ≤ K 且未熔断: 额外结算\n"
        "    (S_T − K) × 累计天数 × (N − 1) 杠杆腿"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DP"): (
        "【每日观察熔断保障累计 ASGQ_DP】\n"
        "每日观察 + 每日结算:\n"
        "  • 未熔断: (S − K), S ≤ K 时乘 N 倍\n"
        "  • 熔断日起: 每日 (S − P)\n\n"
        "比到期版对路径更敏感, Delta/Gamma 跳跃更剧烈."
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DF"): (
        "【每日观察熔断固定赔付累计 ASGQ_DF】\n"
        "  • 未熔断: (S − K), S ≤ K 时 N 倍\n"
        "  • 熔断日起: 每日固定金额 `amount`"
    ),
    ("累计期权 (Decumulator)", "Opt_ASGQ_DFF"): (
        "【每日熔断双固赔累计 ASGQ_DFF】\n"
        "每日观察 + 每日结算, 双固定赔付:\n"
        "  • K < S < H (区间): 每日固定金额 `fix`\n"
        "  • S ≤ K           : (S − K) N 倍\n"
        "  • 熔断日起        : 每日固定金额 `amount`"
    ),
    ("亚式期权 (Asian)", "Asian"): (
        "【亚式期权 Asian】\n"
        "Payoff = clip( mean(S[-N:]) − K,  minPay,  maxPay ) × cp\n\n"
        "  • 取最后 N 个交易日均价与 K 的差额\n"
        "  • minPay / maxPay 限定赔付区间\n"
        "  • 平均化显著降低末日价格风险\n"
        "  • Gamma / Vega 远低于同期限 Vanilla"
    ),
    ("亚式期权 (Asian)", "EnhanceAsian"): (
        "【增强亚式 EnhanceAsian】\n"
        "每日先做价格增强:\n"
        "  • Call: 观察价 = max(S, E)\n"
        "  • Put : 观察价 = min(S, E)\n"
        "再求均值与 K 比较, 并 clip 到 [minPay, maxPay].\n\n"
        "E 提供'每日保底'效果, 提升买方期望."
    ),
    ("气囊期权 (Airbag)", "Opt_Airbag"): (
        "【气囊期权 Opt_Airbag】\n"
        "到期结算, 路径判断是否敲入 KI:\n"
        "  • 未敲入 (Call: min(S) > KI):  pr × max(S_T − K, 0)\n"
        "  • 已敲入                    : pr_ki × (S_T − K)\n\n"
        "小幅下行时买方有软垫保护 (payoff=0 而非负);\n"
        "一旦跌破 KI, 转为线性承担下行, 即'气囊爆掉'."
    ),
    ("雪球期权 (Snowball)", "Opt_Snowball"): (
        "【雪球期权 Opt_Snowball】 (cp=-1 雪球 / cp=1 反雪球)\n"
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
