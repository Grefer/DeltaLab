# _*_ coding: utf-8 _*_
"""bar 级定价记忆化 / 轻量 bump copy / 优选进度回调的回归测试。

这三样都是**纯加速与纯可观测性**的改动，共同的验收标准只有一条：
输出必须逐位不变。所以这里的断言几乎全是 ``array_equal`` 与 ``==``，
而不是 ``pytest.approx``——一旦哪天出现 1e-16 的差，那就是真的有人
把数值动了，不该被容差放过去。

只用合成数据，不依赖 Wind 终端。
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pricing import HedgeBacktest, StrategyCase
from pricing.Option_AB import Option_AB
from pricing.Option_DE import Option_DE
from pricing.Option_SNB import Option_SNB
from pricing.hedge_analysis import (
    _rolling_progress_total,
    _strict_lookback_segment_lengths,
    recommend_by_rolling_history,
)
from pricing.hedge_backtest import CloseToCloseStrategy, SigmaBandStrategy
from pricing.option_base import OptionBase, _memo_key, price_memo

def assert_results_identical(plain, cached):
    """逐位比较两份回测结果的**全部**字段。

    不列白名单：字段直接从结果里枚举，将来引擎多返回一个数组也自动纳入
    比较。这里刻意不用 approx——记忆化若真的改了哪怕 1 ulp，那也是数值被
    动了，必须失败。
    """
    assert set(plain) == set(cached), "结果字段集合不一致"
    for key, left in plain.items():
        right = cached[key]
        if isinstance(left, np.ndarray):
            assert left.shape == right.shape, f"{key} 形状不一致"
            if left.dtype.kind in "fc":
                assert np.array_equal(left, right, equal_nan=True), \
                    f"{key} 数值不一致"
            else:
                assert np.array_equal(left, right), f"{key} 数值不一致"
        elif isinstance(left, float) and np.isnan(left):
            assert isinstance(right, float) and np.isnan(right), f"{key} 不一致"
        else:
            assert left == right, f"{key} 不一致"


def _path(n=42, seed=5, s0=100.0, vol=0.012):
    rng = np.random.default_rng(seed)
    return s0 * np.exp(np.cumsum(rng.normal(0.0, vol, n + 1)))


def _airbag(npath=800, t_days=20):
    return Option_AB("Opt_Airbag", 100.0, [], 100.0, 90.0, t_days,
                     list(range(1, t_days + 1)),
                     0.20, 0.8, 1.0, 1, r=0.03, q=0.03, nPath=npath)


def _snowball(npath=800, t=40):
    return Option_SNB(
        "Opt_Snowball", 100.0, 100.0, 100.0, 80.0, 103.0, t,
        0.18, 0.15, 0.15, 0.2, 1, -1,
        r=0.03, q=0.03, sr=[], ko_observ=[20, t], nPath=npath,
        margin_call=False)


def _decumulator(npath=800, t_days=20):
    return Option_DE("Opt_Decumulator", 100.0, [], 90.0, 0, t_days,
                     list(range(1, t_days + 1)), 0.18, 110.0, 2, 1,
                     r=0.03, q=0.03, nPath=npath)


# --------------------------------------------------------------------------
#  1. 记忆化不改变任何数值
# --------------------------------------------------------------------------

@pytest.mark.parametrize("make_option", [_airbag, _snowball, _decumulator])
@pytest.mark.parametrize("strategy_factory", [
    lambda: CloseToCloseStrategy(),
    lambda: SigmaBandStrategy(k=1.0),
])
def test_price_memo_keeps_backtest_bit_identical(make_option, strategy_factory):
    """开/关记忆化，回测的每个输出数组都必须逐位相同。"""
    prices = _path()

    def run():
        return HedgeBacktest(
            make_option(), prices, strategy=strategy_factory(),
            tc_rate=3e-4, quantity=100.0, multiplier=5.0,
            steps_per_day=1, force_day_close_hedge=True,
        ).run()

    plain = run()
    with price_memo():
        cached = run()

    assert_results_identical(plain, cached)


def test_price_memo_actually_hits():
    """光"结果一样"不够——还得证明它真的省下了定价调用。"""
    option = _airbag()
    prices = _path(n=20)
    calls = {"n": 0}
    real = type(option).get_price

    def counted(self):
        calls["n"] += 1
        return real(self)

    type(option).get_price = counted
    try:
        HedgeBacktest(_airbag(), prices, strategy=CloseToCloseStrategy(),
                      force_day_close_hedge=True).run()
        plain_calls = calls["n"]
        calls["n"] = 0
        with price_memo():
            HedgeBacktest(_airbag(), prices, strategy=CloseToCloseStrategy(),
                          force_day_close_hedge=True).run()
        cached_calls = calls["n"]
    finally:
        type(option).get_price = real

    # 每根调仓 bar 现状是 11 次定价（1 次 bar 价 + get_greeks 内 10 次），
    # 其中 price0 与 bar 价同状态，必然命中；单条回测就该省掉约 1/11。
    assert cached_calls < plain_calls, "记忆化一次都没命中"


def test_price_memo_scope_is_released():
    """作用域退出后不得继续记忆化，否则缓存会跨段留存。"""
    option = _airbag()
    with price_memo() as memo:
        option.priced()
        assert len(memo) == 1
    calls = {"n": 0}
    real = type(option).get_price

    def counted(self):
        calls["n"] += 1
        return real(self)

    type(option).get_price = counted
    try:
        option.priced()
        option.priced()
    finally:
        type(option).get_price = real
    assert calls["n"] == 2, "作用域外仍在读缓存"


def test_price_memo_bypassed_when_seed_is_none():
    """mc_seed=None 的契约是每次重新抽样；缓存必须让路。

    见 option_base 顶部对 mc_seed 的说明：None 表示走 OS 熵。若缓存不做
    例外，两次调用会返回同一个值，把这条语义静默改掉。
    """
    option = _airbag(npath=2000)
    option.mc_seed = None
    with price_memo() as memo:
        first = option.priced()
        second = option.priced()
    assert not memo, "mc_seed=None 时不该写入缓存"
    assert first != second, "mc_seed=None 时两次定价却拿到同一个值"


def test_memo_key_separates_option_subtypes():
    """optiontype 是定价方法的分派键，必须进 key。

    漏掉它的后果是累计期权的 13 个子类型算出同一个 key、互相读到对方的
    价格且不报错——history_bar_cache 的注释里记着这笔学费。
    """
    keys = set()
    for subtype in ("Opt_Decumulator", "Opt_Decumulator_Back",
                    "Opt_EnDecumulator"):
        option = _decumulator()
        option.optiontype = subtype
        keys.add(_memo_key(option))
    assert len(keys) == 3


def test_memo_key_tracks_mutable_state():
    """s0 / sr / nPath / sigma 变了，key 必须变。"""
    base = _snowball()
    key0 = _memo_key(base)
    for attr, value in [("s0", 101.0), ("sr", [100.0]), ("nPath", 400),
                        ("sigma", 0.19), ("mc_seed", 21)]:
        other = _snowball()
        setattr(other, attr, value)
        assert _memo_key(other) != key0, f"{attr} 改变后 key 没变"


def test_memo_key_gives_up_on_unknown_attribute():
    """认不出的属性类型必须整体放弃缓存，而不是猜一个 hash。"""
    option = _airbag()
    option.weird = object()
    assert _memo_key(option) is None
    with price_memo() as memo:
        option.priced()
    assert not memo


# --------------------------------------------------------------------------
#  2. 轻量 bump copy 不共享可变容器
# --------------------------------------------------------------------------

def test_bumped_copy_does_not_share_mutable_containers():
    """父子对象改各自的 sr / observ 互不影响。

    原实现用 deepcopy 保证这点，现在换成"浅拷贝 + 逐个复制容器"，
    这条不变量必须仍然成立。
    """
    option = _decumulator()
    option.sr = [100.0, 101.0]
    clone = option._bumped_copy(s0=99.0)

    assert clone.sr is not option.sr
    assert clone.observ is not option.observ
    clone.sr.append(999.0)
    clone.observ.append(999)
    assert option.sr == [100.0, 101.0]
    assert 999 not in option.observ
    assert clone.s0 == 99.0 and option.s0 == 100.0


def test_bumped_copy_copies_ndarray_fields():
    option = _airbag()
    option.grid = np.arange(4.0)
    clone = option._bumped_copy()
    clone.grid[0] = 42.0
    assert option.grid[0] == 0.0


def test_bumped_copy_prices_identically_to_deepcopy():
    """换掉 deepcopy 之后，bump 出来的价格必须一模一样。"""
    import copy as _copy

    for make in (_airbag, _snowball, _decumulator):
        option = make()
        option.sr = []
        light = option._bumped_copy(s0=103.0).get_price()
        heavy_obj = _copy.deepcopy(option)
        heavy_obj.s0 = 103.0
        assert light == heavy_obj.get_price()


# --------------------------------------------------------------------------
#  3. Option_DE 必填参数校验
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subtype,missing", [
    ("Opt_Decumulator_Fix", "fix"),
    ("Opt_Decumulator_Fix_E", "fix"),
    ("Opt_EnDecumulator_Fix", "fix"),
    ("Opt_ASGQ_call_put", "P"),
    ("Opt_ASGQ_EP", "P"),
    ("Opt_ASGQ_DP", "P"),
    ("Opt_ASGQ_EF", "amount"),
    ("Opt_ASGQ_DF", "amount"),
    ("Opt_ASGQ_EFF", "amount"),
    ("Opt_ASGQ_EFF", "fix"),
    ("Opt_ASGQ_DFF", "amount"),
    ("Opt_ASGQ_DFF", "fix"),
])
def test_decumulator_requires_its_optional_params(subtype, missing):
    """缺参数要在构造期点名报错，而不是在 numpy 深处炸 TypeError。

    GUI 把 fix/P/amount 的默认值设成 0.0，而建构闭包用 ``if p[...]`` 判空，
    0.0 是 falsy 会被转成 None——13 个子类型里 10 个因此开箱即崩。
    """
    # 必须匹配"(fix)"这样的括号形式：直接 match="P" 会被子类型名里的
    # "Opt_ASGQ_EP" / "_DP" 匹配上，把字段映射改错也照样绿。
    with pytest.raises(ValueError, match=re.escape(f"({missing})")):
        Option_DE(subtype, 100.0, [], 90.0, 0, 20, list(range(1, 21)),
                  0.18, 110.0, 2, 1)


@pytest.mark.parametrize("subtype", [
    "Opt_Decumulator", "Opt_Decumulator_Back", "Opt_EnDecumulator",
])
def test_decumulator_without_optional_params_still_builds(subtype):
    """不需要这些参数的子类型不能被误伤。"""
    option = Option_DE(subtype, 100.0, [], 90.0, 0, 20, list(range(1, 21)),
                       0.18, 110.0, 2, 1, nPath=200)
    assert option.get_price() is not None


def test_decumulator_all_subtypes_price_once_params_supplied():
    option_kwargs = dict(nPath=200, fix=2.0, P=95.0, amount=1.5)
    for subtype in Option_DE._REQUIRED_PARAMS:
        option = Option_DE(subtype, 100.0, [], 90.0, 0, 20,
                           list(range(1, 21)), 0.18, 110.0, 2, 1,
                           **option_kwargs)
        assert np.isfinite(option.get_price())


# --------------------------------------------------------------------------
#  4. 优选进度回调
# --------------------------------------------------------------------------

def _history_frame(n=140, seed=9):
    rng = np.random.default_rng(seed)
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0, 0.011, n))),
        index=pd.bdate_range("2023-06-01", periods=n), name="close")


def _selection_cases():
    return [
        StrategyCase("close_to_close", CloseToCloseStrategy(), {}),
        StrategyCase("band_1.0", SigmaBandStrategy(k=1.0), {}),
        StrategyCase("band_1.5", SigmaBandStrategy(k=1.5), {}),
    ]


_SELECTION_KWARGS = dict(
    tc_rate=3e-4, position=1, quantity=100.0, multiplier=5.0,
    slippage_bps=0.0, force_day_close_hedge=True)


def test_progress_callback_is_monotonic_and_exact():
    """分母开跑前就算准，done 从 1 连续走到 total，一个不多一个不少。"""
    history = _history_frame()
    option = _airbag(npath=300)
    cases = _selection_cases()
    lookbacks = {"week": 5, "month": 20, "quarter": 61}

    seen = []
    recommend_by_rolling_history(
        option, history, cases, dict(_SELECTION_KWARGS),
        lookbacks=lookbacks, steps_per_day=1,
        progress_callback=lambda d, t, label, case: seen.append(
            (d, t, label, case)))

    assert seen, "进度回调一次都没被调用"
    totals = {total for _, total, _, _ in seen}
    assert len(totals) == 1, "分母在中途变过"
    total = totals.pop()
    assert [d for d, _, _, _ in seen] == list(range(1, total + 1))

    expected = sum(
        len(_strict_lookback_segment_lengths(days, option.T_days)) * len(cases)
        for days in lookbacks.values()
        if len(history) - days - 1 >= 0)
    assert total == expected
    assert {label for _, _, label, _ in seen} == set(lookbacks)


def test_progress_total_helper_matches_actual_units():
    """分母助手与主循环必须共用同一条证据充足判据。"""
    # 证据不足的档不参与计数：n_history_groups - days - 1 < 0
    total = _rolling_progress_total(
        {"week": 5, "year": 243}, maturity_days=20,
        n_history_groups=100, n_cases=3)
    assert total == len(_strict_lookback_segment_lengths(5, 20)) * 3


def test_progress_callback_failure_does_not_break_selection():
    """进度只是显示用；回调抛异常不该把整轮优选带下水。"""
    history = _history_frame()

    def boom(*_args):
        raise RuntimeError("进度条炸了")

    _recs, ranking, _windows = recommend_by_rolling_history(
        _airbag(npath=300), history, _selection_cases(),
        dict(_SELECTION_KWARGS), lookbacks={"month": 20},
        steps_per_day=1, progress_callback=boom)
    assert not ranking.empty


def test_selection_ranking_is_identical_with_and_without_memo(monkeypatch):
    """整张排名表在开/关记忆化下必须完全相同——这是 A1 能上的唯一标准。"""
    history = _history_frame()
    kwargs = dict(_SELECTION_KWARGS)
    lookbacks = {"week": 5, "month": 20, "quarter": 61}

    def run():
        return recommend_by_rolling_history(
            _airbag(npath=300), history, _selection_cases(), dict(kwargs),
            lookbacks=lookbacks, steps_per_day=1)[1]

    cached = run()
    # 把 priced 打回原形 = 完全不缓存
    monkeypatch.setattr(OptionBase, "priced",
                        lambda self: self.get_price(), raising=True)
    plain = run()

    assert list(plain.columns) == list(cached.columns)
    numeric = plain.select_dtypes(include=[np.number]).columns
    assert np.array_equal(plain[numeric].to_numpy(),
                          cached[numeric].to_numpy(), equal_nan=True)
    other = [c for c in plain.columns if c not in numeric]
    assert plain[other].equals(cached[other])


# --------------------------------------------------------------------------
#  5. 变异测试暴露的三个覆盖缺口
#
#  下面三条各自对应一个"改坏了但全量测试仍全绿"的变异：
#    - 从 _memo_key 里删掉 _intraday_elapsed  → 香草 + 日内平盘会错价
#    - 把 with price_memo() 换成 nullcontext   → 缓存整个失效但没人发现
#    - 给期权加一个进定价的私有字段而忘了进 key → 静默错价
# --------------------------------------------------------------------------

def test_memo_distinguishes_intraday_elapsed():
    """日内 bar 只靠 _intraday_elapsed 区分——它是唯一"漏了就错价"的字段。

    其余字段漏掉最多是 key 摘不出来退回不缓存（fail-safe）；这个私有字段
    是显式补进 key 的，一旦被删，日内价格走平的那些 bar 会共享同一个 key
    并读到错误的剩余期限价值。用香草期权是因为它是唯一在 get_price 里消费
    该字段的结构（见 OptionBase._intraday_elapsed 的说明）。
    """
    from pricing.Option_Vanilla import Option_Vanilla

    option = Option_Vanilla("Eu", 100.0, [], 100.0, 3, 0.2, 1,
                            r=0.03, q=0.03)
    keys = {
        _memo_key(option._bumped_copy(_intraday_elapsed=e))
        for e in (0.0, 0.25, 0.5, 0.75)
    }
    assert len(keys) == 4, "_intraday_elapsed 没有进 key"

    # 端到端：日内价格完全走平，各 bar 只差 _intraday_elapsed。
    prices = np.full(13, 100.0)
    prices[6:] = 101.0

    def run():
        return HedgeBacktest(
            Option_Vanilla("Eu", 100.0, [], 100.0, 3, 0.2, 1,
                           r=0.03, q=0.03),
            prices, strategy=CloseToCloseStrategy(), steps_per_day=4,
            tc_rate=3e-4, quantity=100.0, multiplier=5.0,
        ).run()

    plain = run()
    with price_memo():
        cached = run()
    assert_results_identical(plain, cached)


def test_memo_key_covers_every_pricing_field():
    """守卫：期权对象上不该出现 _memo_key 看不见的字段。

    _memo_key 枚举公开非可调用属性，私有字段只显式收了
    ``_intraday_elapsed``。将来若有人再加一个进定价的私有字段而忘了同步
    这里，本条会红——那正是唯一"漏了就静默错价"的一类改动。
    """
    known_private = {"_intraday_elapsed"}
    for option in (_airbag(), _snowball(), _decumulator()):
        private = {
            name for name in vars(option)
            if name.startswith("_") and not name.startswith("__")
        }
        unexpected = private - known_private
        assert not unexpected, (
            f"{type(option).__name__} 有 _memo_key 覆盖不到的私有字段: "
            f"{sorted(unexpected)}")


def test_selection_actually_uses_the_memo():
    """选优路径上 with price_memo() 必须真的生效。

    只断言"开关缓存结果一样"是不够的——把两处 with 换成 nullcontext 之后
    那条测试照样绿。这里直接数定价调用次数。
    """
    history = _history_frame(n=90)
    cases = _selection_cases()
    lookbacks = {"month": 20}
    calls = {"n": 0}
    real = Option_AB.get_price

    def counted(self):
        calls["n"] += 1
        return real(self)

    Option_AB.get_price = counted
    try:
        recommend_by_rolling_history(
            _airbag(npath=200), history, cases, dict(_SELECTION_KWARGS),
            lookbacks=lookbacks, steps_per_day=1)
        cached_calls = calls["n"]

        calls["n"] = 0
        saved = OptionBase.priced
        OptionBase.priced = lambda self: self.get_price()
        try:
            recommend_by_rolling_history(
                _airbag(npath=200), history, cases, dict(_SELECTION_KWARGS),
                lookbacks=lookbacks, steps_per_day=1)
        finally:
            OptionBase.priced = saved
        plain_calls = calls["n"]
    finally:
        Option_AB.get_price = real

    # 3 个候选共享同一批 bar 级定价，调用次数该降到一半以下。
    assert cached_calls <= plain_calls * 0.5, (
        f"记忆化没在选优路径上生效: {plain_calls} → {cached_calls}")


def test_progress_reporter_reaches_total_with_skips():
    """advance 与 skip 合起来必须恰好走满 total。

    分母按"段数 × 候选数"预算，但有两类段在进入候选循环之前就被 continue
    掉（entry 自带 error、段终端完整性校验失败）。若不补记，进度条会永久
    停在某个百分比——比没有进度条更糟。
    """
    from pricing.hedge_analysis import _ProgressReporter

    seen = []
    reporter = _ProgressReporter(
        lambda d, t, label, case: seen.append((d, t, label, case)), 6)
    reporter.skip(3, "month")          # 一整段被跳过（3 个候选）
    for name in ("a", "b", "c"):
        reporter.advance("quarter", name)

    assert reporter.done == reporter.total == 6
    assert [d for d, _, _, _ in seen] == [3, 4, 5, 6]
    assert seen[-1][0] == seen[-1][1], "进度没走满"


def test_progress_reporter_skip_ignores_non_positive():
    from pricing.hedge_analysis import _ProgressReporter

    seen = []
    reporter = _ProgressReporter(lambda *a: seen.append(a), 2)
    reporter.skip(0)
    reporter.skip(-5)
    assert reporter.done == 0 and not seen


def test_progress_reporter_survives_callback_failure():
    """回调抛异常后应当就地熄火，而不是每个单元再抛一次。"""
    from pricing.hedge_analysis import _ProgressReporter

    calls = {"n": 0}

    def boom(*_args):
        calls["n"] += 1
        raise RuntimeError("nope")

    reporter = _ProgressReporter(boom, 3)
    for name in ("a", "b", "c"):
        reporter.advance("month", name)
    assert calls["n"] == 1, "回调失败后没有停用"
    assert reporter.done == 3


@pytest.mark.parametrize("make_option", [_airbag, _snowball, _decumulator])
def test_memo_key_changes_when_any_public_field_changes(make_option):
    """穷举：改动**任何一个**公开属性，key 都必须变。

    这条守着 _memo_key 的枚举逻辑本身。此前只测了手挑的几个字段，
    把某个字段排除出 key（例如给 dir() 循环加一条跳过规则）能让全量测试
    照样全绿，而后果是两个不同期权共享同一个价格。
    """
    base = make_option()
    key0 = _memo_key(base)
    assert key0 is not None

    names = [
        name for name in dir(base)
        if not name.startswith("_")
        and not callable(getattr(base, name))
    ]
    assert names, "没有枚举到任何公开属性"

    for name in names:
        value = getattr(base, name)
        if isinstance(value, bool):
            mutated = not value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            mutated = type(value)(value + 1)
        elif isinstance(value, str):
            mutated = value + "_x"
        elif isinstance(value, list):
            mutated = list(value) + [1.0]
        elif value is None:
            mutated = 1.0
        else:
            pytest.fail(f"{name} 的类型 {type(value).__name__} 未被覆盖")

        other = make_option()
        setattr(other, name, mutated)
        assert _memo_key(other) != key0, f"改了 {name} 但 key 没变"
