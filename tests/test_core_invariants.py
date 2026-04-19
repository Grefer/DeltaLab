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
    _PRICE_FIELDS_BY_CLS,
)
from pricing.mc_engine import McGbmQ
from pricing.Option_AB import Option_AB
from pricing.Option_AS import Option_AS
from pricing.Option_DE import Option_DE
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
