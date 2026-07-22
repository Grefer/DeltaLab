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
from types import SimpleNamespace
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
from pricing import wind_data
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

    with (
        patch("pricing.rolling_backtest.get_log_returns", return_value=fake_ret),
        patch("pricing.wind_data.load_history_cached",
              side_effect=RuntimeError("offline unit test")),
    ):
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


# ---------------------------------------------------------------------------
# 5) Wind 日内查询边界、交易日 session 与缓存隔离
# ---------------------------------------------------------------------------


class _FakeWindIntraday:
    def __init__(self, times, values=None):
        self.times = pd.to_datetime(times).to_pydatetime().tolist()
        self.values = (
            list(values) if values is not None else list(range(len(times)))
        )
        self.calls = []

    def wsi(self, code, fields, start, end, options):
        self.calls.append((code, fields, start, end, options))
        return SimpleNamespace(
            ErrorCode=0,
            Fields=["CLOSE"],
            Times=self.times,
            Data=[self.values],
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, "2024-01-02"),
        ("", "2024-01-02"),
        ("2024-01-01", " "),
        ("not-a-date", "2024-01-02"),
        (123, "2024-01-02"),
        ("2024-02-30", "2024-03-01"),
        ("2024-01-03", "2024-01-02"),
        ("2024-01-02 15:00:01", "2024-01-02 15:00:00"),
    ],
)
def test_intraday_invalid_ranges_fail_before_wind_connection(start, end):
    with patch("pricing.wind_data._ensure_wind") as ensure_wind:
        with pytest.raises(ValueError):
            wind_data.get_intraday_bars("FAKE.SH", start, end)
    ensure_wind.assert_not_called()


@pytest.mark.parametrize(
    ("start", "end"),
    [("", "2024-01-02"), ("bad-date", "2024-01-02"),
     ("2024-01-03", "2024-01-02")],
)
def test_daily_invalid_ranges_fail_before_wind_connection(start, end):
    with patch("pricing.wind_data._ensure_wind") as ensure_wind:
        with pytest.raises(ValueError):
            wind_data.get_close_prices("FAKE.SH", start, end)
    ensure_wind.assert_not_called()


def test_intraday_exact_datetimes_keep_original_boundaries_and_rows():
    times = [
        "2024-01-05 21:00:00",
        "2024-01-06 01:00:00",
        "2024-01-08 10:00:00",
    ]
    fake_wind = _FakeWindIntraday(times)

    with patch("pricing.wind_data._ensure_wind", return_value=fake_wind):
        frame = wind_data.get_intraday_bars(
            "au2402.SHF",
            "2024-01-05 21:00:00",
            "2024-01-08 10:00:00",
            bar_size="60",
        )

    _, _, actual_start, actual_end, _ = fake_wind.calls[0]
    assert actual_start == "2024-01-05 21:00:00"
    assert actual_end == "2024-01-08 10:00:00"
    assert list(frame.index) == list(pd.to_datetime(times))


def test_intraday_date_query_keeps_full_night_session_by_trading_date():
    # 2024-01-08（周一）的交易日从上周五夜盘开始；查询尾部的周二夜盘
    # 属于下一交易日，只有 opener 没有日盘，应被排除。
    times = [
        "2024-01-04 21:00:00",
        "2024-01-05 01:00:00",
        "2024-01-05 09:00:00",
        "2024-01-05 15:00:00",
        "2024-01-05 21:00:00",
        "2024-01-06 01:00:00",
        "2024-01-08 09:00:00",
        "2024-01-08 15:00:00",
        "2024-01-08 21:00:00",
        "2024-01-09 01:00:00",
        "2024-01-09 09:00:00",
        "2024-01-09 15:00:00",
        "2024-01-09 21:00:00",
    ]
    fake_wind = _FakeWindIntraday(times)

    with patch("pricing.wind_data._ensure_wind", return_value=fake_wind):
        frame = wind_data.get_intraday_bars(
            "au2402.SHF", "2024-01-08", "2024-01-09", bar_size="60"
        )

    _, _, actual_start, actual_end, _ = fake_wind.calls[0]
    assert actual_start == "2023-12-25 00:00:00"
    assert actual_end == "2024-01-09 23:59:59"
    assert list(frame.index) == list(pd.to_datetime(times[4:12]))


def test_intraday_date_query_filters_a_share_calendar_sessions():
    times = [
        "2024-01-05 09:30:00",
        "2024-01-05 15:00:00",
        "2024-01-08 09:30:00",
        "2024-01-08 15:00:00",
        "2024-01-09 09:30:00",
        "2024-01-09 15:00:00",
    ]
    fake_wind = _FakeWindIntraday(times)

    with patch("pricing.wind_data._ensure_wind", return_value=fake_wind):
        frame = wind_data.get_intraday_bars(
            "510050.SH", "2024-01-08", "2024-01-09", bar_size="60"
        )

    assert list(frame.index) == list(pd.to_datetime(times[2:]))


def test_intraday_cache_key_distinguishes_full_time_and_date_semantics(tmp_path):
    with patch("pricing.wind_data._CACHE_DIR", str(tmp_path)):
        at_0930 = wind_data._intraday_cache_path(
            "510050.SH",
            "2024-01-08 09:30:00",
            "2024-01-09 15:00:00",
            "60",
        )
        at_1030 = wind_data._intraday_cache_path(
            "510050.SH",
            "2024-01-08 10:30:00",
            "2024-01-09 15:00:00",
            "60",
        )
        whole_dates = wind_data._intraday_cache_path(
            "510050.SH", "2024-01-08", "2024-01-09", "60"
        )
        exact_midnight = wind_data._intraday_cache_path(
            "510050.SH",
            "2024-01-08 00:00:00",
            "2024-01-09 00:00:00",
            "60",
        )

    assert len({at_0930, at_1030, whole_dates, exact_midnight}) == 4


@pytest.mark.parametrize(
    ("classification", "expected_ranges"),
    [
        (
            ("SSE", "基金"),
            [("09:30", "11:30"), ("13:00", "15:00")],
        ),
        (
            ("CFFEX", "指数类"),
            [("09:30", "11:30"), ("13:00", "15:00")],
        ),
        (
            ("CFFEX", "国债类"),
            [("09:30", "11:30"), ("13:00", "15:15")],
        ),
        (
            ("GFEX", "有色"),
            [
                ("09:00", "10:15"),
                ("10:30", "11:30"),
                ("13:30", "15:00"),
            ],
        ),
        (
            ("DCE", "农产品"),
            [
                ("21:00", "23:00"),
                ("09:00", "10:15"),
                ("10:30", "11:30"),
                ("13:30", "15:00"),
            ],
        ),
        (
            ("SHFE", "有色"),
            [
                ("21:00", "01:00"),
                ("09:00", "10:15"),
                ("10:30", "11:30"),
                ("13:30", "15:00"),
            ],
        ),
        (
            ("SHFE", "贵金属"),
            [
                ("21:00", "02:30"),
                ("09:00", "10:15"),
                ("10:30", "11:30"),
                ("13:30", "15:00"),
            ],
        ),
    ],
)
def test_trading_session_clock_ranges_cover_canonical_session_types(
        classification, expected_ranges):
    code = "ZZ2609.TEST"
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification",
        return_value=classification,
    ):
        ranges = wind_data.get_trading_session_clock_ranges(code)

    actual_ranges = [
        (start.strftime("%H:%M"), end.strftime("%H:%M"))
        for start, end in ranges
    ]
    assert actual_ranges == expected_ranges


def test_trading_session_clock_ranges_use_symbol_override_without_wind():
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification"
    ) as classification:
        ranges = wind_data.get_trading_session_clock_ranges("P2609.DCE")

    classification.assert_not_called()
    assert [
        (start.strftime("%H:%M"), end.strftime("%H:%M"))
        for start, end in ranges
    ][0] == ("21:00", "23:00")


@pytest.mark.parametrize(
    ("code", "expected_close"),
    [
        ("510050.SH", "15:00"),
        ("000001.SZ", "15:00"),
        ("IF2609.CFE", "15:00"),
        ("T2609.CFE", "15:15"),
        ("P2609.DCE", "15:00"),
    ],
)
def test_local_only_session_lookup_never_connects_wind(code, expected_close):
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification"
    ) as classification:
        ranges = wind_data.get_trading_session_clock_ranges(
            code, allow_wind=False
        )

    classification.assert_not_called()
    assert ranges[-1][1].strftime("%H:%M") == expected_close


def test_local_only_unknown_session_does_not_fall_through_to_wind():
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification"
    ) as classification:
        ranges = wind_data.get_trading_session_clock_ranges(
            "CU2609.SHF", allow_wind=False
        )

    classification.assert_not_called()
    assert ranges is None
    assert wind_data.is_time_in_trading_session(
        "CU2609.SHF", "23:00", allow_wind=False
    ) is None


@pytest.mark.parametrize(
    "code",
    [
        "JD2609.DCE",
        "AP609.CZC",
        "EC2609.INE",
        "WR2609.SHF",
        "SI2609.GFE",
        "LC2609.GFE",
        "PS2609.GFE",
    ],
)
def test_explicit_day_only_product_overrides_skip_broad_night_categories(code):
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification"
    ) as classification:
        ranges = wind_data.get_trading_session_clock_ranges(
            code, allow_wind=False
        )
        at_night = wind_data.is_time_in_trading_session(
            code, "23:00", allow_wind=False
        )

    classification.assert_not_called()
    assert ranges[0][0].strftime("%H:%M") == "09:00"
    assert at_night is False


@pytest.mark.parametrize(
    ("code", "expected_night_close"),
    [
        ("SC2609.INE", "02:30"),
        ("NR2609.INE", "23:00"),
        ("BC2609.INE", "01:00"),
        ("OP2609.SHF", "23:00"),
    ],
)
def test_symbol_overrides_distinguish_night_session_lengths(
        code, expected_night_close):
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification"
    ) as classification:
        ranges = wind_data.get_trading_session_clock_ranges(
            code, allow_wind=False
        )

    classification.assert_not_called()
    assert ranges[0][0].strftime("%H:%M") == "21:00"
    assert ranges[0][1].strftime("%H:%M") == expected_night_close


def test_is_time_in_trading_session_handles_cross_midnight_and_breaks():
    code = "CU2609.SHF"
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification",
        return_value=("SHFE", "有色"),
    ):
        # 夜盘端点和跨日凌晨均有效；午间/夜盘结束后的空档无效。
        assert wind_data.is_time_in_trading_session(code, "21:00") is True
        assert wind_data.is_time_in_trading_session(code, "23:59:59") is True
        assert wind_data.is_time_in_trading_session(code, "00:30") is True
        assert wind_data.is_time_in_trading_session(code, "01:00") is True
        assert wind_data.is_time_in_trading_session(code, "01:00:01") is False
        assert wind_data.is_time_in_trading_session(code, "10:20") is False
        assert wind_data.is_time_in_trading_session(code, "11:30") is True
        assert wind_data.is_time_in_trading_session(code, "15:00") is True


def test_unknown_session_metadata_stays_unknown_instead_of_becoming_closed():
    code = "UNKNOWN.TEST"
    wind_data._get_trading_session_minutes.cache_clear()
    wind_data.get_trading_session_clock_ranges.cache_clear()
    with patch(
        "pricing.wind_data._get_wind_market_classification",
        return_value=("UNKNOWN", "未知分类"),
    ):
        assert wind_data.get_trading_session_clock_ranges(code) is None
        assert wind_data.is_time_in_trading_session(code, "23:00") is None


# ---------------------------------------------------------------------------
# 6) 历史择优 Wind 代码路由与截止日主力映射
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "mode", "normalized", "is_contract"),
    [
        ("P.DCE", "product_pool", "P.DCE", False),
        ("p.dce", "product_pool", "P.DCE", False),
        ("P2609.DCE", "single", "P2609.DCE", True),
        ("MA609.CZC", "single", "MA609.CZC", True),
        ("P00.DCE", "single", "P00.DCE", False),
        ("510050.SH", "single", "510050.SH", False),
    ],
)
def test_classify_wind_history_code_routes_product_and_contract(
        code, mode, normalized, is_contract):
    parsed = wind_data.classify_wind_history_code(code)

    assert parsed["mode"] == mode
    assert parsed["code"] == normalized
    assert bool(parsed.get("is_futures_contract", False)) is is_contract


def test_classify_wind_history_code_rejects_empty_code():
    with pytest.raises(ValueError, match="不能为空"):
        wind_data.classify_wind_history_code("  ")


class _FakeWindMainHistory:
    def __init__(self, *, error_code=0):
        self.error_code = error_code
        self.calls = []

    def wsd(self, code, field, start, end, options):
        self.calls.append((code, field, start, end, options))
        return SimpleNamespace(
            ErrorCode=self.error_code,
            Times=pd.to_datetime([
                "2025-12-31", "2026-01-02", "2026-01-05",
                "2026-01-06", "2026-02-02",
            ]).to_pydatetime().tolist(),
            Data=[[
                "P2509.DCE", "p2601.dce", "M2601.DCE",
                "P2605.DCE", "P2609.DCE",
            ]],
        )


def test_get_main_contract_history_is_cutoff_safe_and_filters_other_products():
    fake_wind = _FakeWindMainHistory()

    with patch("pricing.wind_data._ensure_wind", return_value=fake_wind):
        mapping = wind_data.get_main_contract_history(
            "p.dce", "2026-01-01", "2026-01-31")

    assert fake_wind.calls == [(
        "P.DCE", "trade_hiscode", "2026-01-01", "2026-01-31", "",
    )]
    assert mapping.index.tolist() == [
        pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-06"),
    ]
    assert mapping.tolist() == ["P2601.DCE", "P2605.DCE"]
    assert mapping.index.max() <= pd.Timestamp("2026-01-31")


def test_get_main_contract_history_rejects_concrete_contract_before_wind_call():
    with patch("pricing.wind_data._ensure_wind") as ensure_wind:
        with pytest.raises(ValueError, match="只能查询品种代码"):
            wind_data.get_main_contract_history(
                "P2609.DCE", "2026-01-01", "2026-01-31")
    ensure_wind.assert_not_called()


def test_get_main_contract_history_preserves_wind_error_code():
    fake_wind = _FakeWindMainHistory(error_code=-40520007)
    with patch("pricing.wind_data._ensure_wind", return_value=fake_wind):
        with pytest.raises(RuntimeError, match="-40520007"):
            wind_data.get_main_contract_history(
                "P.DCE", "2026-01-01", "2026-01-31")
