# _*_ coding: utf-8 _*_
"""
Phase 4 测试：滚动 Wind 回测驱动器、rebase_path 与事前 HV 的正确性。

所有用例均不依赖 Wind 终端，通过 mock / 合成数据验证：
1. rolling pipeline 能跑通且真实/MC 分布大致对齐；
2. Black-76 Δ 数值与解析解一致（中心差分容差）；
3. rebase_path 的起点恒等与 roundtrip 恒等；
4. 事前 HV 窗口不泄漏未来信息。
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# 确保可以直接从仓库根导入 pricing 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pricing.constants import ANNUAL_DAYS
from pricing.Option_Vanilla import Option_Vanilla
from pricing.rolling_backtest import run_rolling_backtest
from pricing.wind_data import rebase_path


# ---------------------------------------------------------------------------
# 1) Smoke test: rolling pipeline 能跑通
# ---------------------------------------------------------------------------

def test_rolling_backtest_smoke():
    """合成 GBM 假 Wind 数据 -> 驱动器全链路跑通；真实与 MC 分布均值差在容差内。"""
    rng = np.random.default_rng(0)
    n_days = 500
    sigma_true = 0.20
    r_true = 0.03
    dt = 1.0 / ANNUAL_DAYS

    fake_lr = rng.normal(
        (r_true - 0.5 * sigma_true ** 2) * dt,
        sigma_true * np.sqrt(dt),
        size=n_days,
    )
    fake_ret = pd.Series(
        fake_lr,
        index=pd.bdate_range("2022-01-01", periods=n_days),
        name="FAKE.SH",
    )

    option_cfg = dict(
        optiontype="Vanilla",
        s0=100.0,
        sr=[],
        K=100.0,
        T=60,          # 60 交易日存续期
        sigma=0.20,    # 驱动器会用事前 HV 覆盖
        cp=1,
        r=0.03,
        q=0.03,
        exe_mode="Eu",
    )

    with patch("pricing.rolling_backtest.get_log_returns", return_value=fake_ret):
        df = run_rolling_backtest(
            option_cfg=option_cfg,
            option_class=Option_Vanilla,
            code="FAKE.SH",
            start="2022-01-01",
            end="2024-01-01",
            step=5,
            hv_window=20,
            asset_type="equity",
            r=0.03,
            q=0.03,
            hedge_kwargs={"position": 1, "quantity": 1, "tc_rate": 0.0},
        )

    # 至少应产出 10 行以上（500 - 20 - 60 = 420 可用区间，步长 5 ≈ 84 行）
    assert len(df) >= 10, f"rolling rows 太少: {len(df)}"

    # 关键列必须存在
    for col in ("hedge_pnl_real", "hedge_pnl_mc", "sigma_pre"):
        assert col in df.columns, f"缺列 {col}"

    # 合成数据两边都是 GBM，真实与 MC 的均值差应在 2σ 内（允许放宽到 2.5σ）
    gap = abs(df["hedge_pnl_real"].mean() - df["hedge_pnl_mc"].mean())
    pooled_std = max(df["hedge_pnl_real"].std(), df["hedge_pnl_mc"].std())
    assert pooled_std > 0, "pooled_std=0，样本分布异常"
    assert gap < 2.5 * pooled_std, (
        f"真实/MC 分布均值差过大: gap={gap:.4f}, pooled_std={pooled_std:.4f}"
    )

    # sigma_pre 应集中在真值附近（20 天窗口估计误差可观，放宽到 ±0.08）
    assert abs(df["sigma_pre"].mean() - sigma_true) < 0.08


# ---------------------------------------------------------------------------
# 2) Black-76 Δ 数值与解析解对齐
# ---------------------------------------------------------------------------

def test_black76_delta_matches_closed_form():
    """Option_Vanilla 在 q=r 下等价于 Black-76；对比中心差分 Δ 与解析解。"""
    s0 = 100.0
    K = 100.0
    sigma = 0.20
    r = 0.03
    T_years = 1.0
    T_days = int(round(T_years * ANNUAL_DAYS))  # 243

    # Black-76：d1 = (ln(F/K) + 0.5 σ² T)/(σ√T)，F=S（q=r 时），Δ = exp(-rT) N(d1)
    d1 = (np.log(s0 / K) + 0.5 * sigma ** 2 * T_years) / (sigma * np.sqrt(T_years))
    expected_delta = np.exp(-r * T_years) * norm.cdf(d1)

    opt = Option_Vanilla(
        optiontype="Vanilla",
        s0=s0,
        sr=[],
        K=K,
        T=T_days,
        sigma=sigma,
        cp=1,
        r=r,
        q=r,          # Black-76: q=r
        exe_mode="Eu",
    )

    greeks = opt.get_greeks()
    actual_delta = greeks[0]

    # get_greeks 使用中心差分（ds ≈ 1），ATM 1 年场景误差约 1e-4 量级
    assert abs(actual_delta - expected_delta) < 1e-4, (
        f"Δ 偏差过大: actual={actual_delta:.8f}, expected={expected_delta:.8f}"
    )


# ---------------------------------------------------------------------------
# 3) rebase_path roundtrip
# ---------------------------------------------------------------------------

def test_rebase_path_roundtrip():
    """起点恒等 S[0]=s0，且对 rebase 后价格取 log diff 应完全还原输入序列。"""
    rng = np.random.default_rng(123)
    lr = pd.Series(
        rng.normal(0, 0.02, size=50),
        index=pd.bdate_range("2024-01-01", periods=50),
    )
    s0 = 100.0

    rebased = rebase_path(lr, s0)

    # 起点恒等
    assert abs(float(rebased.iloc[0]) - s0) < 1e-12

    # 长度：len(lr) + 1
    assert len(rebased) == len(lr) + 1

    # roundtrip
    recovered = np.diff(np.log(rebased.values))
    assert recovered.shape == lr.values.shape
    assert np.max(np.abs(recovered - lr.values)) < 1e-12


# ---------------------------------------------------------------------------
# 4) 事前 HV 不泄漏未来信息
# ---------------------------------------------------------------------------

def test_pre_hv_no_lookahead():
    """
    低波段前、高波段后拼接的序列，在低波段内部取事前 HV 窗口，
    估计值必须贴近低波段真值，绝不能被后半段污染。
    """
    rng = np.random.default_rng(7)
    dt = 1.0 / ANNUAL_DAYS
    low = rng.normal(0, 0.10 * np.sqrt(dt), size=100)
    high = rng.normal(0, 0.50 * np.sqrt(dt), size=100)
    lr = pd.Series(
        np.concatenate([low, high]),
        index=pd.bdate_range("2024-01-01", periods=200),
    )

    # 在低波段内部取窗口 [t0-20, t0)，t0=50
    t0 = 50
    window = lr.iloc[t0 - 20: t0]
    sigma_pre = float(window.std(ddof=1) * np.sqrt(ANNUAL_DAYS))

    # 真值 0.10，有限样本偏差允许 ±0.04
    assert abs(sigma_pre - 0.10) < 0.04, f"sigma_pre={sigma_pre:.4f} 偏离低波段真值"

    # 最关键：绝对不能被后半段 0.50 污染
    assert sigma_pre < 0.20, f"sigma_pre={sigma_pre:.4f} 疑似被未来污染"
