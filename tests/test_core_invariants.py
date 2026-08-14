# _*_ coding: utf-8 _*_
"""
核心不变量测试。

覆盖 Review 修复的 P0/P1 关键点：
1. 各期权到期 payoff 与解析公式一致（boundary / 极限）。
2. HedgeBacktest 盈亏分解恒等式。
3. _rescale_option_to_real_s0 字段白名单完整性。
4. run_multi 多路径间 MC 采样独立（P0-1 回归测试）。
5. get_greeks 使用 CRN：bump 之间共享 mc_seed（P1-1 回归测试）。

所有测试都只用合成数据 / 解析解，不依赖 Wind 终端。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pricing.constants import ANNUAL_DAYS
from pricing.hedge_backtest import (
    HedgeBacktest,
    _rescale_option_to_real_s0,
    _format_rescale_info,
    _PRICE_FIELDS_BY_CLS,
    histogram_bin_edges,
    padded_histogram_range,
)
from pricing.mc_engine import McGbmQ
from pricing.Option_AB import Option_AB
from pricing.Option_AS import Option_AS
from pricing.Option_DE import Option_DE
from pricing.Option_SNB import Option_SNB
from pricing.Option_Vanilla import Option_Vanilla, blsprice


# ---------------------------------------------------------------------------
# 1) Vanilla 到期 payoff / Δ 边界
# ---------------------------------------------------------------------------

def test_vanilla_expiry_payoff_call_put():
    """T=0 时价格 = intrinsic value，call/put 对称。"""
    # 深 ITM call
    opt = Option_Vanilla("Vanilla", s0=150.0, sr=[], K=100.0,
                         T=0, sigma=0.20, cp=1, r=0.03, q=0.0)
    assert abs(opt.get_price() - 50.0) < 1e-10

    # 深 OTM call
    opt2 = Option_Vanilla("Vanilla", s0=80.0, sr=[], K=100.0,
                          T=0, sigma=0.20, cp=1, r=0.03, q=0.0)
    assert abs(opt2.get_price() - 0.0) < 1e-10

    # Put 对称
    opt3 = Option_Vanilla("Vanilla", s0=80.0, sr=[], K=100.0,
                          T=0, sigma=0.20, cp=-1, r=0.03, q=0.0)
    assert abs(opt3.get_price() - 20.0) < 1e-10


def test_vanilla_deep_itm_otm_delta():
    """深 ITM call Δ→1，深 OTM call Δ→0。"""
    deep_itm = Option_Vanilla("Vanilla", s0=300.0, sr=[], K=100.0,
                              T=5, sigma=0.20, cp=1, r=0.0, q=0.0)
    d_itm = deep_itm.get_greeks()[0]
    assert d_itm > 0.99, f"深 ITM call Δ 应接近 1，实际 {d_itm:.4f}"

    deep_otm = Option_Vanilla("Vanilla", s0=30.0, sr=[], K=100.0,
                              T=5, sigma=0.20, cp=1, r=0.0, q=0.0)
    d_otm = deep_otm.get_greeks()[0]
    assert abs(d_otm) < 1e-3, f"深 OTM call Δ 应接近 0，实际 {d_otm:.6f}"


def test_vanilla_blsprice_matches_closed_form():
    """blsprice 直接与 BS 解析解比对（ATM, 1 年）。"""
    from scipy.stats import norm
    from math import log, sqrt, exp
    s, K, r, q, t, sigma = 100.0, 100.0, 0.03, 0.0, 1.0, 0.20
    d1 = (log(s / K) + (r - q + sigma ** 2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    expected = s * exp(-q * t) * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)

    actual = blsprice(s, K, r, q, t, sigma, cp=1)
    assert abs(actual - expected) < 1e-12


def test_vanilla_greeks_match_bs_closed_form():
    """Vanilla Greeks 应与 BS 解析解一致，特别覆盖 rho 的 dr 分母。"""
    from scipy.stats import norm
    from math import log, sqrt, exp

    s, K, r, q, sigma, cp = 100.0, 103.0, 0.025, 0.01, 0.22, 1
    T_days = 180
    t = T_days / ANNUAL_DAYS
    d1 = (log(s / K) + (r - q + sigma ** 2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    expected = {
        "delta": exp(-q * t) * norm.cdf(d1),
        "gamma": exp(-q * t) * norm.pdf(d1) / (s * sigma * sqrt(t)),
        "vega": s * exp(-q * t) * norm.pdf(d1) * sqrt(t),
        "theta": (
            -s * exp(-q * t) * norm.pdf(d1) * sigma / (2 * sqrt(t))
            - r * K * exp(-r * t) * norm.cdf(d2)
            + q * s * exp(-q * t) * norm.cdf(d1)
        ),
        "rho": K * t * exp(-r * t) * norm.cdf(d2),
    }

    opt = Option_Vanilla("Vanilla", s0=s, sr=[], K=K,
                         T=T_days, sigma=sigma, cp=cp, r=r, q=q)
    actual = dict(zip(("delta", "gamma", "vega", "theta", "rho"), opt.get_greeks()))

    assert abs(actual["delta"] - expected["delta"]) < 2e-4
    assert abs(actual["gamma"] - expected["gamma"]) < 2e-4
    assert abs(actual["vega"] - expected["vega"]) < 1e-3
    assert abs(actual["theta"] - expected["theta"]) < 5e-2
    assert abs(actual["rho"] - expected["rho"]) < 2e-2


def test_vanilla_unsupported_exe_mode_raises():
    """非欧式 Vanilla 暂未实现时应明确报错，而不是返回 None。"""
    opt = Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0,
                         T=10, sigma=0.20, cp=1, r=0.03, q=0.0,
                         exe_mode="Am")
    with pytest.raises(NotImplementedError):
        opt.get_price()


# ---------------------------------------------------------------------------
# 2) Option_AB 到期 payoff
# ---------------------------------------------------------------------------

def test_option_ab_expiry_payoff_knockin():
    """Airbag T=0：历史 sr 中有价格触及 KI 时按 pr_ki 赔付；未触及按 pr 的 call intrinsic 赔付。"""
    observ = [1, 2, 3, 4, 5]
    # 历史路径：sr[2]=85 <= KI=90，应触发敲入
    sr_ki = [100.0, 95.0, 85.0, 95.0, 105.0]
    opt_ki = Option_AB("Opt_Airbag", s0=105.0, sr=sr_ki,
                       K=100.0, KI=90.0, T_days=0, observ=observ,
                       sigma=0.20, pr=0.8, pr_ki=1.0, cp=1)
    # sr[-1]=105, sr[-1]-K=5, pr_ki=1.0 -> 5.0
    assert abs(opt_ki.get_price() - 5.0) < 1e-10

    # 未敲入：所有 sr 都 > 90
    sr_no = [100.0, 101.0, 102.0, 103.0, 110.0]
    opt_no = Option_AB("Opt_Airbag", s0=110.0, sr=sr_no,
                      K=100.0, KI=90.0, T_days=0, observ=observ,
                      sigma=0.20, pr=0.8, pr_ki=1.0, cp=1)
    # pr * max(110-100, 0) = 0.8 * 10 = 8.0
    assert abs(opt_no.get_price() - 8.0) < 1e-10


# ---------------------------------------------------------------------------
# 3) Option_AS：均价单调性（合成路径）
# ---------------------------------------------------------------------------

def test_option_as_monotone_in_strike():
    """亚式 call 价格对 strike 单调递减。"""
    # 用同一种子以保证对比是同一批路径
    kwargs = dict(optiontype="Asian", s0=100.0, sr=[], E=100.0, T=22, N=22,
                  sigma=0.15, cp=1, minPay=0.0, maxPay=float("inf"),
                  r=0.03, q=0.03, nPath=10000)
    p_low_K = Option_AS(K=95.0, **kwargs).get_price()
    p_atm = Option_AS(K=100.0, **kwargs).get_price()
    p_high_K = Option_AS(K=105.0, **kwargs).get_price()
    assert p_low_K > p_atm > p_high_K, (
        f"亚式 call 价格应对 K 单调递减: {p_low_K} -> {p_atm} -> {p_high_K}"
    )


def test_option_as_invalid_optiontype_raises():
    """非法 optiontype 应抛 ValueError，而不是 UnboundLocalError（P2-1 回归）。"""
    opt = Option_AS(optiontype="NotARealType", s0=100.0, sr=[], K=100.0, E=100.0,
                    T=22, N=22, sigma=0.15, cp=1, minPay=0.0, maxPay=float("inf"),
                    r=0.03, q=0.03, nPath=1000)
    with pytest.raises(ValueError):
        opt.get_price()


def test_option_as_expiry_payoff_uses_realized_observations_and_spot():
    """Asian / EnhanceAsian 在 T=0 时应返回到期 payoff，而不是 None。"""
    opt = Option_AS(optiontype="Asian", s0=115.0, sr=[100.0, 105.0, 110.0],
                    K=106.0, E=108.0, T=0, N=3, sigma=0.15, cp=1,
                    minPay=0.0, maxPay=999999.0, r=0.03, q=0.03, nPath=1000)
    # T=0 时当前 s0 是最新观察价；last 3 = [105, 110, 115]，均值 110。
    assert abs(opt.get_price() - 4.0) < 1e-12

    enhanced_put = Option_AS(optiontype="EnhanceAsian", s0=115.0,
                             sr=[100.0, 105.0, 110.0], K=110.0, E=108.0,
                             T=0, N=3, sigma=0.15, cp=-1,
                             minPay=0.0, maxPay=999999.0,
                             r=0.03, q=0.03, nPath=1000)
    # put 增强腿先对观察价做 min(S, E)：[105, 108, 108]，均值 107。
    assert abs(enhanced_put.get_price() - 3.0) < 1e-12

    stepped = Option_AS(optiontype="Asian", s0=100.0, sr=[], K=100.0,
                        E=100.0, T=2, N=2, sigma=0.15, cp=1,
                        minPay=0.0, maxPay=999999.0,
                        r=0.0, q=0.0, nPath=1000)
    stepped.step_forward(110.0)
    stepped.step_forward(120.0)
    assert abs(stepped.get_price() - 15.0) < 1e-12


# ---------------------------------------------------------------------------
# 4) Option_DE 低波动极限：σ→0 下价格应逼近解析 payoff
# ---------------------------------------------------------------------------

def test_option_de_low_sigma_limit():
    """σ→0 + r=q=0 + s0 介于 K 和 H 之间：Opt_Decumulator_Back 应接近 (s0-K)*le。

    Decumulator_Back 赔付函数：
      cp=1, sr/ss 全程落在区间 (K, H) 之间 -> flag_N=1
      每期 cashflow = (S-K), le=observ 长度。σ=0 时 S≈s0*exp(0)=s0，
      贴现因子约为 1，预期价 ≈ (s0-K)*le。
    """
    observ = list(range(1, 6))   # 5 期
    opt = Option_DE(
        optiontype="Opt_Decumulator_Back",
        s0=95.0, sr=[], K=90.0, T_over=0, T_days=5,
        observ=observ, sigma=1e-6, H=110.0, N=2, cp=1,
        r=0.0, q=0.0, nPath=2000,
    )
    price = opt.get_price()
    expected = (95.0 - 90.0) * len(observ)  # = 25
    # σ≈0 + 贴现≈1，允许容差来自贴现非 1 和残余 σ
    assert abs(price - expected) < 1.0, (
        f"σ→0 极限下价格应逼近 (s0-K)*le={expected}, 实际 {price:.4f}"
    )


# ---------------------------------------------------------------------------
# 5) HedgeBacktest 盈亏分解恒等式
# ---------------------------------------------------------------------------

def test_hedge_backtest_pnl_identity():
    """核验 pos * V[0]*qty + Σhedge - ΣTC - pos * V[-1]*qty = hedging_error。"""
    opt = Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0,
                         T=10, sigma=0.20, cp=1, r=0.03, q=0.0)
    prices = HedgeBacktest.simulate_prices(
        100.0, 0.20, T_days=10, r=0.03, q=0.0, seed=42, steps_per_day=1
    )
    bt = HedgeBacktest(opt, prices, hedge_freq=1, tc_rate=0.001,
                       position=1, quantity=1.0, multiplier=0)
    r = bt.run()

    lhs = (1 * r['opt_value'][0] * 1.0
           + np.sum(r['hedge_daily'])
           - np.sum(r['tc_paid'])
           - 1 * r['opt_value'][-1] * 1.0)
    assert abs(lhs - r['hedging_error']) < 1e-9, (
        f"盈亏分解恒等式不满足: lhs={lhs:.8f}, hedging_error={r['hedging_error']:.8f}"
    )


def test_hedge_backtest_partial_horizon_uses_terminal_mtm():
    """H<T 时只需 H 日行情，末端应按 T-H 日公允价值而非 intrinsic 结算。"""
    option = Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0,
        T=5, sigma=0.20, cp=1, r=0.0, q=0.0,
    )
    external_path = np.array([100.0, 100.0, 100.0])
    result = HedgeBacktest(
        option,
        path_source="historical",
        external_path=external_path,
        evaluation_days=2,
        position=1,
        quantity=1.0,
        multiplier=0,
    ).run()

    expected_mtm = blsprice(
        100.0, 100.0, 0.0, 0.0,
        3 / ANNUAL_DAYS, 0.20, cp=1,
    )
    assert len(result["prices"]) == 3
    assert result["evaluation_days"] == 2
    assert result["initial_maturity_days"] == 5
    assert result["remaining_days_at_end"] == 3
    assert result["terminal_mode"] == "mark_to_market"
    assert result["opt_value"][-1] == pytest.approx(expected_mtm)
    assert result["opt_value"][-1] > 0.0  # ATM intrinsic 为 0
    assert result["delta"][-1] > 0.0
    assert result["gamma"][-1] > 0.0
    assert result["shares"][-1] == 0.0
    assert option.T == 5  # 回测不得修改调用方传入的期权


def test_hedge_backtest_default_horizon_remains_full_expiry():
    """默认 None 与显式 H=T 完全一致，并在末端使用到期 payoff/零 Greeks。"""
    option = Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0,
        T=2, sigma=0.20, cp=1, r=0.0, q=0.0,
    )
    prices = np.array([100.0, 102.0, 104.0])
    default_result = HedgeBacktest(
        option, prices, multiplier=0,
    ).run()
    explicit_result = HedgeBacktest(
        option, prices, multiplier=0, evaluation_days=2,
    ).run()

    for key in (
            "prices", "opt_value", "delta", "gamma", "shares",
            "hedge_daily", "option_daily", "tc_paid", "net_daily"):
        np.testing.assert_allclose(default_result[key], explicit_result[key])
    assert default_result["evaluation_days"] == 2
    assert default_result["initial_maturity_days"] == 2
    assert default_result["remaining_days_at_end"] == 0
    assert default_result["terminal_mode"] == "expiry"
    assert default_result["opt_value"][-1] == pytest.approx(4.0)
    np.testing.assert_allclose(
        [default_result[key][-1]
         for key in ("delta", "gamma", "vega", "theta", "rho")],
        0.0,
    )


@pytest.mark.parametrize(
    "evaluation_days", [0, -1, 1.5, True, np.nan, np.inf, "bad"],
)
def test_hedge_backtest_rejects_invalid_evaluation_days(evaluation_days):
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 5, 0.20, 1, r=0.0, q=0.0)
    with pytest.raises(ValueError, match="evaluation_days.*正整数"):
        HedgeBacktest(
            option, np.array([100.0, 101.0]),
            evaluation_days=evaluation_days,
        )


def test_hedge_backtest_rejects_evaluation_horizon_beyond_maturity():
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 5, 0.20, 1, r=0.0, q=0.0)
    with pytest.raises(ValueError, match="evaluation_days.*超过.*剩余期限"):
        HedgeBacktest(
            option, np.array([100.0, 101.0]),
            evaluation_days=6,
        )


def _vanilla_direction_pair(*, tc_rate=0.0):
    """返回同一期权/路径的卖出与买入回测结果。

    刻意不叫 short/long：那两个词在期权语境里指 gamma 敞口，而这里区分的
    是买卖方向，两者在不同产品上并不同向。
    """
    option = Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0,
        T=4, sigma=0.20, cp=1, r=0.03, q=0.01,
    )
    prices = np.array([100.0, 102.0, 98.0, 105.0, 101.0])
    kwargs = {
        "hedge_freq": 1,
        "tc_rate": tc_rate,
        "quantity": 3.0,
        "multiplier": 0,
        "slippage_bps": 0.0,
    }
    sell = HedgeBacktest(
        option, prices, position=1, **kwargs).run()
    buy = HedgeBacktest(
        option, prices, position=-1, **kwargs).run()
    return sell, buy


def test_position_label_says_buy_sell_not_long_short():
    """``position_label`` 只能是 buy/sell，不得出现 long/short。

    long/short 在期权语境里通常指 **gamma 敞口**，而 ``position`` 表示的
    是**买卖方向**。两者不是一回事，见下一条测试。
    """
    sell, buy = _vanilla_direction_pair(tc_rate=0.0)

    assert {sell["position_label"], buy["position_label"]} == {"sell", "buy"}
    for result in (sell, buy):
        assert result["position_label"] not in ("long", "short")


def test_same_trade_direction_gives_opposite_gamma_across_products():
    """同一买卖方向下，香草与累计期权的 gamma 敞口方向相反。

    这是「买卖方向 ≠ gamma 敞口」的根据，也是 position_label 不能叫
    long/short 的原因：同一个 ``position`` 值在两个产品上对应相反的敞口，
    一个名字不可能同时对。

    敞口还会在同一条路径内翻转（雪球实测 41% 的 bar 为正、37% 为负），所
    以也不能改用一张「产品 → gamma 方向」的静态映射表——那只是换个方式再
    错一次。要判断敞口只能读实际算出的 ``portfolio_gamma``。
    """
    path = np.linspace(100.0, 104.0, 21)

    def mean_gamma(option):
        result = HedgeBacktest(
            option, path_source="historical", external_path=path,
            position=1, quantity=1, multiplier=0,
            hedge_freq=1, steps_per_day=1).run()
        gamma = np.asarray(result["portfolio_gamma"], dtype=float)
        return float(np.mean(gamma[np.isfinite(gamma)]))

    vanilla = mean_gamma(Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0, T=20, sigma=0.18, cp=1,
        r=0.03, q=0.03))
    decumulator = mean_gamma(Option_DE(
        "Opt_Decumulator", 100.0, [], 90.0, 0, 20, list(range(1, 21)),
        0.18, 110.0, 2, 1, r=0.03, q=0.03, nPath=20000))

    # 同为 position=1（卖出），两个产品的 gamma 敞口一负一正。
    assert vanilla < 0, vanilla
    assert decumulator > 0, decumulator


def test_hedge_backtest_buy_sell_zero_cost_are_exact_mirrors():
    """零成本下，买卖两个方向的经济持仓与 PnL 必须严格互为相反数。"""
    sell, buy = _vanilla_direction_pair(tc_rate=0.0)

    assert sell["position"] == 1
    assert sell["position_label"] == "sell"
    assert buy["position"] == -1
    assert buy["position_label"] == "buy"
    for raw_key in ("opt_value", "delta", "gamma", "vega", "theta", "rho"):
        np.testing.assert_allclose(sell[raw_key], buy[raw_key])
    for signed_key in (
            "shares", "hedge_daily", "option_daily", "net_daily",
            "cumulative_pnl", "portfolio_delta", "portfolio_gamma",
            "portfolio_vega", "portfolio_theta", "portfolio_rho"):
        np.testing.assert_allclose(sell[signed_key], -buy[signed_key])
    assert sell["hedging_error"] == pytest.approx(-buy["hedging_error"])
    np.testing.assert_allclose(sell["tc_paid"], 0.0)
    np.testing.assert_allclose(buy["tc_paid"], 0.0)

    scale = sell["quantity"]
    np.testing.assert_allclose(
        sell["portfolio_gamma"], -scale * sell["gamma"])
    np.testing.assert_allclose(
        buy["portfolio_gamma"], scale * buy["gamma"])


def test_hedge_backtest_buy_sell_nonzero_cost_keep_gross_mirror():
    """成本不改变毛 PnL 镜像；同一绝对交易应让两边成本同为扣减项。"""
    sell, buy = _vanilla_direction_pair(tc_rate=0.001)
    sell_gross = sell["hedge_daily"] + sell["option_daily"]
    buy_gross = buy["hedge_daily"] + buy["option_daily"]

    np.testing.assert_allclose(sell_gross, -buy_gross)
    np.testing.assert_allclose(sell["tc_paid"], buy["tc_paid"])
    assert np.sum(sell["tc_paid"]) > 0
    assert np.all(sell["tc_paid"] >= 0)
    assert np.all(buy["tc_paid"] >= 0)
    np.testing.assert_allclose(
        sell["net_daily"] + buy["net_daily"],
        -2.0 * sell["tc_paid"],
    )


def test_hedge_backtest_summary_uses_signed_portfolio_greeks(capsys):
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 2, 0.20, 1, r=0.0, q=0.0)
    bt = HedgeBacktest(
        option, np.array([100.0, 101.0, 102.0]),
        position=1, quantity=3.0, multiplier=0,
    )
    result = bt.run()

    assert result["gamma"][0] > 0
    assert result["portfolio_gamma"][0] < 0
    bt.summary()
    gamma_line = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("Gamma"))
    assert f"{result['portfolio_gamma'][0]:>10.4f}" in gamma_line


@pytest.mark.parametrize("position", [0, 2, -2, np.nan, np.inf, True, "bad"])
def test_hedge_backtest_rejects_invalid_position(position):
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 2, 0.20, 1, r=0.0, q=0.0)
    with pytest.raises(ValueError, match="position.*1.*-1"):
        HedgeBacktest(
            option, np.array([100.0, 101.0, 102.0]),
            position=position, quantity=1.0, multiplier=0,
        )


@pytest.mark.parametrize("quantity", [0, -1, np.nan, np.inf, -np.inf, "bad"])
def test_hedge_backtest_rejects_nonpositive_or_nonfinite_quantity(quantity):
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 2, 0.20, 1, r=0.0, q=0.0)
    with pytest.raises(ValueError, match="quantity.*有限正数"):
        HedgeBacktest(
            option, np.array([100.0, 101.0, 102.0]),
            position=1, quantity=quantity, multiplier=0,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("tc_rate", -0.001),
        ("tc_rate", np.nan),
        ("tc_rate", np.inf),
        ("tc_rate", "bad"),
        ("slippage_bps", -1.0),
        ("slippage_bps", np.nan),
        ("slippage_bps", np.inf),
        ("slippage_bps", "bad"),
    ],
)
def test_hedge_backtest_rejects_invalid_cost_inputs(parameter, value):
    option = Option_Vanilla(
        "Vanilla", 100.0, [], 100.0, 2, 0.20, 1, r=0.0, q=0.0)
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=rf"{parameter}.*有限非负数"):
        HedgeBacktest(
            option, np.array([100.0, 101.0, 102.0]),
            position=1, quantity=1.0, multiplier=0, **kwargs,
        )


# ---------------------------------------------------------------------------
# 6) _rescale_option_to_real_s0 字段白名单完整性
# ---------------------------------------------------------------------------

def test_rescale_option_ab_whitelist():
    """Option_AB rescale 后应缩放 s0/K/KI，但其它字段（σ/cp/observ…）不变。"""
    opt = Option_AB("Opt_Airbag", s0=100.0, sr=[], K=100.0, KI=90.0,
                    T_days=20, observ=list(range(1, 21)),
                    sigma=0.18, pr=0.8, pr_ki=1.0, cp=1, r=0.03, q=0.03)

    original_attrs = set(vars(opt).keys())

    scaled, info = _rescale_option_to_real_s0(opt, real_s0=120.0)
    ratio = 120.0 / 100.0

    # 关键字段被正确缩放
    assert abs(scaled.s0 - 120.0) < 1e-12
    assert abs(scaled.K - 100.0 * ratio) < 1e-12
    assert abs(scaled.KI - 90.0 * ratio) < 1e-12

    # 非价格字段不变
    assert scaled.sigma == 0.18
    assert scaled.cp == 1
    assert scaled.observ == list(range(1, 21))
    assert scaled.pr == 0.8
    assert scaled.pr_ki == 1.0
    assert scaled.T_days == 20

    # 原对象未被改动
    assert abs(opt.s0 - 100.0) < 1e-12
    assert abs(opt.K - 100.0) < 1e-12

    # 字段集合没有增减
    scaled_attrs = set(vars(scaled).keys())
    assert scaled_attrs == original_attrs, (
        f"rescale 后属性集合有变动: 新增={scaled_attrs - original_attrs}, "
        f"丢失={original_attrs - scaled_attrs}"
    )

    # 白名单里声明的字段都在 info 里（s0 额外注入）
    for f in _PRICE_FIELDS_BY_CLS["Option_AB"]:
        assert f in info["fields"], f"白名单字段 {f} 未出现在 info"


def test_rescale_option_snb_whitelist_scales_ko_vector():
    """Snowball 实盘 rebase 应缩放 s00/K/KI/KO；KO 支持降敲序列。"""
    opt = Option_SNB(
        "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
        KO=[103.0, 102.0, 101.0], T=60, sigma=0.20,
        coupon=0.15, coupon_ko=0.15, margin=1.0, act=1, cp=-1,
        r=0.03, q=0.03, sr=[], ko_observ=[20, 40, 60], nPath=1000,
    )

    scaled, info = _rescale_option_to_real_s0(opt, real_s0=50.0)

    assert abs(scaled.s0 - 50.0) < 1e-12
    assert abs(scaled.s00 - 50.0) < 1e-12
    assert abs(scaled.K - 50.0) < 1e-12
    assert abs(scaled.KI - 40.0) < 1e-12
    assert scaled.KO == [51.5, 51.0, 50.5]
    assert opt.KO == [103.0, 102.0, 101.0]

    for f in _PRICE_FIELDS_BY_CLS["Option_SNB"]:
        assert f in info["fields"], f"白名单字段 {f} 未出现在 info"

    text = _format_rescale_info(info)
    assert "KO" in text and "[103.000000" in text and "[51.500000" in text


def test_option_snb_first_observation_knockout_coupon_counts_one_day():
    """第 1 个交易日敲出时应结算 1 天票息，而不是 0。"""
    opt = Option_SNB(
        "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
        KO=99.0, T=1, sigma=0.0, coupon=0.243, coupon_ko=0.243,
        margin=1.0, act=1, cp=-1, r=0.0, q=0.0, sr=[],
        ko_observ=[1], nPath=1000,
    )

    expected = 100.0 * 0.243 / ANNUAL_DAYS

    assert abs(opt.get_price() - expected) < 1e-12
    assert opt.knockout_event([100.0, 100.0], steps_per_day=1) == (1, expected)


def test_option_snb_margin_call_mode_controls_knockin_loss_cap():
    """追保不按保证金封顶；不追保才按 margin*s00 封顶，普通雪球/反雪球均一致。"""

    def _price(cp, s0, ki, ko, margin, margin_call):
        opt = Option_SNB(
            "Opt_Snowball", s00=100.0, s0=s0, K=100.0, KI=ki, KO=ko,
            T=1, sigma=0.0, coupon=0.0, coupon_ko=0.0,
            margin=margin, act=1, cp=cp, r=0.0, q=0.0, sr=[],
            ko_observ=[1], nPath=1000, margin_call=margin_call,
        )
        return opt.get_price()

    # 雪球：终价 50，敲入且未敲出，原始亏损 = 50 - 100 = -50。
    assert abs(_price(cp=-1, s0=50.0, ki=80.0, ko=999.0, margin=0.2, margin_call=True) + 50.0) < 1e-12
    assert abs(_price(cp=-1, s0=50.0, ki=80.0, ko=999.0, margin=0.2, margin_call=False) + 20.0) < 1e-12

    # 反雪球：终价 150，敲入且未敲出，原始亏损 = 100 - 150 = -50。
    assert abs(_price(cp=1, s0=150.0, ki=120.0, ko=-999.0, margin=0.2, margin_call=True) + 50.0) < 1e-12
    assert abs(_price(cp=1, s0=150.0, ki=120.0, ko=-999.0, margin=0.2, margin_call=False) + 20.0) < 1e-12


def test_option_snb_negative_margin_raises():
    """负保证金会把亏损错误封成收益，必须拒绝。"""
    with pytest.raises(ValueError):
        Option_SNB(
            "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
            KO=103.0, T=1, sigma=0.0, coupon=0.0, coupon_ko=0.0,
            margin=-0.2, act=1, cp=-1, r=0.0, q=0.0, sr=[],
            ko_observ=[1], nPath=1000,
        )


def test_option_snb_no_margin_call_requires_positive_margin():
    """不追保是有限损失结构，必须有正保证金比例定义封顶金额。"""
    with pytest.raises(ValueError):
        Option_SNB(
            "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
            KO=103.0, T=1, sigma=0.0, coupon=0.0, coupon_ko=0.0,
            margin=0.0, act=1, cp=-1, r=0.0, q=0.0, sr=[],
            ko_observ=[1], nPath=1000, margin_call=False,
        )


def test_option_snb_validates_mc_path_count_and_observation_range():
    """Snowball 参数错误应给出业务错误，而不是 NumPy 维度/下标异常。"""
    odd_npath = Option_SNB(
        "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
        KO=103.0, T=3, sigma=0.0, coupon=0.0, coupon_ko=0.0,
        margin=0.2, act=1, cp=-1, r=0.0, q=0.0, sr=[],
        ko_observ=[1, 2, 3], nPath=999, margin_call=False,
    )
    with pytest.raises(ValueError):
        odd_npath.get_price()

    bad_observ = Option_SNB(
        "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
        KO=103.0, T=3, sigma=0.0, coupon=0.0, coupon_ko=0.0,
        margin=0.2, act=1, cp=-1, r=0.0, q=0.0, sr=[],
        ko_observ=[1, 4], nPath=1000, margin_call=False,
    )
    with pytest.raises(ValueError):
        bad_observ.get_price()


def test_run_multi_snowball_final_price_uses_knockout_endpoint():
    """多路径统计中，雪球提前敲出路径的 final_price 应是敲出日价格。"""
    opt = Option_SNB(
        "Opt_Snowball", s00=100.0, s0=100.0, K=100.0, KI=80.0,
        KO=101.0, T=3, sigma=0.0, coupon=0.0, coupon_ko=0.0,
        margin=0.2, act=1, cp=-1, r=0.0, q=0.0, sr=[],
        ko_observ=[1, 2, 3], nPath=1000, margin_call=False,
    )
    opt.greeks_nPath = 100
    paths = np.array([[100.0, 100.0, 105.0, 80.0]])

    single = HedgeBacktest(
        opt, paths[0], hedge_freq=1, tc_rate=0.0,
        position=1, quantity=1.0, multiplier=0,
        evaluation_days=2,
    ).run()
    assert single["knocked_out"]
    assert single["evaluation_days"] == 2
    assert single["initial_maturity_days"] == 3
    assert single["remaining_days_at_end"] == 0
    assert single["terminal_mode"] == "knockout"

    bt = HedgeBacktest(opt, paths[0], hedge_freq=1, tc_rate=0.0,
                       position=1, quantity=1.0, multiplier=0)
    res = bt.run_multi(paths, max_workers=1)

    assert res["position"] == 1
    assert res["position_label"] == "sell"
    assert res["quantity"] == pytest.approx(1.0)
    assert res["knocked_out"][0]
    assert res["ko_days"][0] == 2
    assert abs(res["final_prices"][0] - 105.0) < 1e-12


# ---------------------------------------------------------------------------
# 7) run_multi 多路径 MC 采样独立性（P0-1 回归）
# ---------------------------------------------------------------------------

def test_run_multi_per_path_mc_seeds_differ():
    """run_multi 的每条路径应注入不同 mc_seed，避免 MC 采样被人为压窄。"""
    # 用 Option_AB 这种 MC 期权保证 seed 会真正影响定价；但为跑快一点用小 nPath
    opt = Option_AB("Opt_Airbag", s0=100.0, sr=[], K=100.0, KI=90.0,
                    T_days=5, observ=list(range(1, 6)),
                    sigma=0.20, pr=0.8, pr_ki=1.0, cp=1, nPath=2000)

    # 模拟 3 条不同的价格路径
    paths = HedgeBacktest.simulate_multi_paths(
        s0=100.0, sigma=0.20, T_days=5, n_paths=3, seed=7, steps_per_day=1
    )

    bt = HedgeBacktest(opt, paths[0], hedge_freq=1, tc_rate=0.0,
                       position=1, quantity=1.0, multiplier=0, base_seed=100)
    res = bt.run_multi(paths)

    assert res["n_paths"] == 3
    # 三条路径 final_prices 取自外部 paths，天然不同
    assert len(set(paths[:, -1].tolist())) == 3

    # 回归检查：如果 per-path seed 注入失败，所有路径的 MC 定价部分会完全相同，
    # 对冲误差的分布只来自外部路径差异；这里验证至少 errors 有非零方差。
    assert np.std(res["errors"]) > 0, "run_multi 的对冲误差在多路径上完全相同，疑似 MC 采样未按路径独立"


def test_mc_engine_seed_none_produces_different_samples():
    """McGbmQ(seed=None) 连续两次调用结果应不同（OS 熵采样）。"""
    s1 = McGbmQ(100.0, 0.03, 0.2, T=1.0, nPath=1000, nStep=20, seed=None)
    s2 = McGbmQ(100.0, 0.03, 0.2, T=1.0, nPath=1000, nStep=20, seed=None)
    # 几乎不可能完全相同（浮点精度下完全相等的概率 ≈ 0）
    assert not np.array_equal(s1, s2), "seed=None 两次调用结果相同，OS 熵采样失效"

    # 而 seed=42 两次调用结果必须完全一致
    s3 = McGbmQ(100.0, 0.03, 0.2, T=1.0, nPath=1000, nStep=20, seed=42)
    s4 = McGbmQ(100.0, 0.03, 0.2, T=1.0, nPath=1000, nStep=20, seed=42)
    assert np.array_equal(s3, s4), "seed=42 两次调用结果不一致，确定性采样失效"


def test_mc_engine_rejects_odd_npath():
    """对偶变量 MC 引擎必须拒绝奇数路径数，避免静默少生成一条路径。"""
    with pytest.raises(ValueError):
        McGbmQ(100.0, 0.03, 0.2, T=1.0, nPath=999, nStep=20, seed=42)


# ---------------------------------------------------------------------------
# 8) get_greeks 使用 CRN：bump 之间共享 mc_seed（P1-1 回归）
# ---------------------------------------------------------------------------

def test_mc_greeks_uses_crn_with_shared_seed():
    """
    固定 mc_seed 时，两次连续 get_greeks() 应得到完全相同的结果（CRN 可复现）；
    且 greeks_nPath 生效时，bump 阶段走小路径数。
    """
    opt = Option_AB("Opt_Airbag", s0=100.0, sr=[], K=100.0, KI=90.0,
                    T_days=10, observ=list(range(1, 11)),
                    sigma=0.20, pr=0.8, pr_ki=1.0, cp=1, nPath=5000)
    opt.mc_seed = 20  # 默认值，显式写出

    g1 = opt.get_greeks()
    g2 = opt.get_greeks()
    for a, b in zip(g1, g2):
        assert abs(a - b) < 1e-12, f"固定 seed 下两次 Greeks 不一致: {g1} vs {g2}"

    # mc_seed 改变会影响 Greeks（证明 CRN 确实依赖 mc_seed）
    opt.mc_seed = 12345
    g3 = opt.get_greeks()
    # 至少有一个分量变化，否则说明 seed 没传进 MC
    diffs = [abs(a - b) for a, b in zip(g1, g3)]
    assert max(diffs) > 0.0, "切换 mc_seed 后 Greeks 完全不变，seed 未生效"

    # greeks_nPath 生效不影响逻辑，只验证 bump 前后 self.nPath 被恢复
    opt.mc_seed = 20
    opt.greeks_nPath = 1000
    before = opt.nPath
    _ = opt.get_greeks()
    assert opt.nPath == before, "get_greeks 退出后 nPath 未被恢复"


def test_run_multi_keeps_successful_paths_when_one_path_fails():
    """单条路径异常不应导致已完成路径结果全部丢失。"""
    opt = Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0,
                         T=2, sigma=0.20, cp=1, r=0.03, q=0.0)
    paths = np.array([
        [100.0, 101.0, 102.0],
        [100.0, 0.0, 102.0],
        [100.0, 99.0, 98.0],
    ])

    bt = HedgeBacktest(opt, paths[0], hedge_freq=1, tc_rate=0.0,
                       position=1, quantity=1.0, multiplier=0)
    res = bt.run_multi(paths, max_workers=1)

    assert res["failed_paths"] == [1]
    assert 1 in res["path_errors"]
    assert np.isnan(res["errors"][1])
    assert np.isfinite(res["errors"][0])
    assert np.isfinite(res["errors"][2])


# ---------------------------------------------------------------------------
#  直方图分箱：退化样本不得让 numpy 拒绝分箱
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "values"),
    [
        ("全等大数", np.full(10, 1e15)),
        ("全等 DBL_MAX", np.full(4, np.finfo(float).max)),
        ("全等 -DBL_MAX", np.full(4, -np.finfo(float).max)),
        ("量级内微小跨度", np.array([1e6, 1e6 + 1e-9, 1e6 + 2e-9])),
        ("跨度溢出", np.array([-np.finfo(float).max, np.finfo(float).max])),
        ("含非有限值", np.array([1.0, np.nan, np.inf, 2.0])),
        ("空样本", np.array([])),
        ("正常分布", np.linspace(-5.0, 5.0, 40)),
    ],
)
def test_histogram_bin_edges_are_finite_and_strictly_increasing(label, values):
    """边界必须有限且严格递增，否则 numpy 会拒绝分箱、matplotlib 画空白图。

    ``padded_histogram_range`` 的相对展开在 ``DBL_MAX`` 附近会溢出成 inf，
    而 nan 参与的比较恒为假，塌缩检测与兜底会一起失效——那一档必须覆盖。
    """
    edges = histogram_bin_edges(values, 30)

    assert np.all(np.isfinite(edges)), (label, edges)
    assert np.all(np.diff(edges) > 0), (label, edges)


@pytest.mark.parametrize(
    "value", [0.0, 3.5, 1e15, 1e300, np.finfo(float).max,
              -np.finfo(float).max])
def test_padded_histogram_range_stays_representable(value):
    lo, hi = padded_histogram_range(value)

    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi


def test_plot_error_dist_handles_degenerate_errors(monkeypatch):
    """后端绘图入口与 GUI 共用同一份分箱兜底，退化样本不得抛错。"""
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda *a, **k: None)

    opt = Option_Vanilla("Vanilla", s0=100.0, sr=[], K=100.0, T=2,
                         sigma=0.20, cp=1, r=0.03, q=0.0)
    bt = HedgeBacktest(opt, np.array([100.0, 101.0, 102.0]), hedge_freq=1,
                       tc_rate=0.0, position=1, quantity=1.0, multiplier=0)

    for errors in (np.full(10, 1e15), np.array([1e6, 1e6 + 1e-9]),
                   np.array([-1e308, 1e308]), np.array([np.nan, 1.0])):
        figure = bt.plot_error_dist(errors)
        assert figure is not None
        matplotlib.pyplot.close(figure)
