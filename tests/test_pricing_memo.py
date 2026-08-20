# _*_ coding: utf-8 _*_
"""bar 级定价记忆化 / 轻量 bump copy / 优选进度与中止的回归测试。

加速与可观测性的改动，共同的验收标准只有一条：输出必须逐位不变。
所以**同一进程内**两种写法的对拍一律用 ``array_equal`` / ``==``——那里
出现 1e-16 的差就是真的有人动了数值，不该被容差放过去。

例外是钉在常量上的黄金值（``_DE_GOLDEN``）：它跨机器比对，而 numpy 的
``exp`` / ``cumsum`` 在不同平台走不同的 SIMD 与 libm，末位会差 1~2 ULP。
那一组用 1e-12 的相对容差，理由见该测试的 docstring。

只用合成数据，不依赖 Wind 终端。
"""
from __future__ import annotations

import os
import pathlib
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
    segment_cost_weight,
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
    """**真的省略**这些参数时要在构造期点名报错，而不是在 numpy 深处炸。

    注意判据是 ``is None``，不是 falsy：0 是合法取值（区间赔付 0 = 区间内
    那几天不结算，熔断赔付 0 = 熔断后不再有现金流）。只有直接调用类却漏传
    才会走到这里——GUI 表单里这三项恒有值。
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
        progress_callback=seen.append)

    assert seen, "进度回调一次都没被调用"
    totals = {p.total for p in seen}
    assert len(totals) == 1, "分母在中途变过"
    total = totals.pop()
    assert [p.done for p in seen] == list(range(1, total + 1))

    expected = sum(
        len(_strict_lookback_segment_lengths(days, option.T_days)) * len(cases)
        for days in lookbacks.values()
        if len(history) - days - 1 >= 0)
    assert total == expected
    assert {p.label for p in seen} == set(lookbacks)

    # 权重分母同样不能中途变；成本份额单调不减、终点恰好 1。
    assert len({round(p.total_weight, 9) for p in seen}) == 1
    fractions = [p.fraction for p in seen]
    assert fractions == sorted(fractions), "成本份额出现回退"
    assert fractions[-1] == pytest.approx(1.0, abs=1e-9)


def test_progress_total_helper_matches_actual_units():
    """分母助手与主循环必须共用同一条证据充足判据。"""
    # 证据不足的档不参与计数：n_history_groups - days - 1 < 0
    total, weight = _rolling_progress_total(
        {"week": 5, "year": 243}, maturity_days=20,
        n_history_groups=100, n_cases=3)
    lengths = _strict_lookback_segment_lengths(5, 20)
    assert total == len(lengths) * 3
    assert weight == pytest.approx(
        3 * sum(segment_cost_weight(x, 20) for x in lengths))


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
    reporter = _ProgressReporter(seen.append, 6)
    reporter.skip(3, "month")          # 一整段被跳过（3 个候选）
    for name in ("a", "b", "c"):
        reporter.advance("quarter", name)

    assert reporter.done == reporter.total == 6
    assert [p.done for p in seen] == [3, 4, 5, 6]
    assert seen[-1].done == seen[-1].total, "进度没走满"
    # 不给权重时退回按单元数计，份额同样要走满。
    assert seen[-1].fraction == pytest.approx(1.0)


def test_progress_reporter_skip_ignores_non_positive():
    from pricing.hedge_analysis import _ProgressReporter

    seen = []
    reporter = _ProgressReporter(seen.append, 2)
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


# --------------------------------------------------------------------------
#  6. 优选可中断
# --------------------------------------------------------------------------

def test_cancel_event_stops_selection_promptly():
    """置位中止标志后，优选必须在下一个单元边界抛 SelectionCancelled。

    "边界"是段与候选的交界——那里没有半写完的累加器。单个
    HedgeBacktest.run() 内部不检查：中断会留下半截状态，而一个单元最多
    几秒。所以判据是"很快停"，不是"立刻停"。
    """
    import threading

    from pricing.hedge_analysis import SelectionCancelled

    history = _history_frame(n=140)
    cases = _selection_cases()
    cancel = threading.Event()
    seen = []

    def cb(progress):
        seen.append(progress.done)
        if progress.done == 3:        # 跑够 3 个单元后请求停止
            cancel.set()

    with pytest.raises(SelectionCancelled):
        recommend_by_rolling_history(
            _airbag(npath=200), history, cases, dict(_SELECTION_KWARGS),
            lookbacks={"week": 5, "month": 20, "quarter": 61},
            steps_per_day=1, progress_callback=cb, cancel_event=cancel)

    # 置位后最多再跑一个单元就该停：advance 在单元开头查标志。
    assert max(seen) <= 4, f"置位后又跑了 {max(seen) - 3} 个单元"


def test_cancel_event_set_upfront_stops_immediately():
    import threading

    from pricing.hedge_analysis import SelectionCancelled

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SelectionCancelled):
        recommend_by_rolling_history(
            _airbag(npath=200), _history_frame(n=140), _selection_cases(),
            dict(_SELECTION_KWARGS), lookbacks={"month": 20},
            steps_per_day=1, cancel_event=cancel)


def test_selection_without_cancel_event_is_unaffected():
    """不传 cancel_event 时行为完全不变。"""
    _recs, ranking, _w = recommend_by_rolling_history(
        _airbag(npath=200), _history_frame(n=140), _selection_cases(),
        dict(_SELECTION_KWARGS), lookbacks={"month": 20}, steps_per_day=1)
    assert not ranking.empty


def test_worker_treats_cancel_as_non_error():
    """worker 收到 SelectionCancelled 要走中止收尾，不能弹错误框。"""
    from types import SimpleNamespace

    from deltalab_ui import runner as runner_mod
    from deltalab_ui.runner import RunnerMixin

    calls = {"cancelled": 0, "failed": []}

    def cancel_now(*_args, **_kwargs):
        raise runner_mod.SelectionCancelled("stop")

    fake = SimpleNamespace(
        # 用户在取行情阶段就按了停止。
        _build_backtest=cancel_now,
        _history_cancel_event=None,
        after=lambda _delay, callback: callback(),
        _finish_job=lambda *a, **k: calls.__setitem__(
            "cancelled", calls["cancelled"] + 1),
        _fail_history_recommendation=calls["failed"].append,
        _make_history_progress_callback=lambda *a: None,
    )
    state = {
        "source": "csv", "csv_path": "x.csv", "csv_col": "close",
        "cfg": {"build": lambda _st, _p: SimpleNamespace(_time_remaining=2)},
        "subtype": "t", "params": {},
        "history_lookbacks": {"month": 20},
    }

    RunnerMixin._history_recommendation_worker(fake, state)

    assert calls["failed"] == [], "中止被当成了错误"
    assert calls["cancelled"] == 1, "没有走中止收尾"


# --------------------------------------------------------------------------
#  7. 分段并行
# --------------------------------------------------------------------------

def test_parallel_segments_match_serial_bit_for_bit():
    """并行与串行的排名表必须逐位相同。

    归并按 plan 顺序串行做，所以 strategy_types 的后写覆盖、
    failure_reasons 的顺序、label_results 的插入顺序都不受调度影响。
    """
    import pricing.hedge_analysis as ha

    history = _history_frame(n=170)
    cases = _selection_cases()
    lookbacks = {"week": 5, "month": 20, "quarter": 61}

    def run():
        return ha.recommend_by_rolling_history(
            _airbag(npath=400), history, cases, dict(_SELECTION_KWARGS),
            lookbacks=lookbacks, steps_per_day=1)[1]

    parallel = run()
    real = ha._selection_max_workers
    ha._selection_max_workers = lambda option, n_units: 1
    try:
        serial = run()
    finally:
        ha._selection_max_workers = real

    assert list(serial.columns) == list(parallel.columns)
    numeric = serial.select_dtypes(include=[np.number]).columns
    assert np.array_equal(serial[numeric].to_numpy(),
                          parallel[numeric].to_numpy(), equal_nan=True)
    other = [c for c in serial.columns if c not in numeric]
    assert serial[other].equals(parallel[other])


def test_map_segments_preserves_plan_order():
    """先算完的不能先归并——顺序是归并语义的前提。"""
    import time

    from pricing.hedge_analysis import _map_segments

    def run_one(plan):
        # 故意让后面的先算完
        time.sleep(0.02 * (3 - plan))
        return plan

    got = _map_segments(run_one, [1, 2, 3], _airbag(npath=200))
    assert got == [1, 2, 3]


def test_map_segments_propagates_first_failure():
    """异常按 plan 顺序抛，与串行执行一致。"""
    from pricing.hedge_analysis import _map_segments

    def run_one(plan):
        if plan in (2, 3):
            raise ValueError(f"boom-{plan}")
        return plan

    with pytest.raises(ValueError, match="boom-2"):
        _map_segments(run_one, [1, 2, 3], _airbag(npath=200))


def test_worker_count_backs_off_on_memory():
    """线程数按单次定价峰值内存收敛，不是按核数开满。

    累计期权 nPath=1e5、T=243 一次定价峰值约 0.8 GB，开满会把机器拖垮
    ——实测 10 线程比串行还慢 2.5 倍。
    """
    from pricing.hedge_analysis import _selection_max_workers

    # 不跟机器核数较劲：把预算调到只够一个线程，验的是"内存这一维真的在
    # 参与决策"，而不是本机恰好有几个核。
    light = _decumulator(npath=2000, t_days=20)
    heavy = _decumulator(npath=100000, t_days=61)
    assert _selection_max_workers(heavy, 8) <= _selection_max_workers(light, 8)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv("DELTALAB_SELECTION_MEM_BUDGET_MB", "64")
        assert _selection_max_workers(heavy, 8) == 1, "内存预算没有生效"
        monkeypatch.setenv("DELTALAB_SELECTION_MEM_BUDGET_MB", "65536")
        assert _selection_max_workers(heavy, 8) > 1, "预算放宽后仍不并行"
        # 旋钮写坏了要退回默认并发，而不是把整轮优选炸掉。
        for bad in ("2g", "", "-1"):
            monkeypatch.setenv("DELTALAB_SELECTION_MEM_BUDGET_MB", bad)
            assert _selection_max_workers(heavy, 8) >= 1
    finally:
        monkeypatch.undo()

    assert _selection_max_workers(light, 1) == 1, "只有一段时不该起线程池"


def test_progress_total_still_exact_under_parallelism():
    """并行下 done 的自增有竞态；加锁后仍必须恰好走满。"""
    history = _history_frame(n=170)
    cases = _selection_cases()
    lookbacks = {"week": 5, "month": 20, "quarter": 61}
    seen = []

    recommend_by_rolling_history(
        _airbag(npath=300), history, cases, dict(_SELECTION_KWARGS),
        lookbacks=lookbacks, steps_per_day=1,
        progress_callback=lambda p: seen.append((p.done, p.total)))

    total = seen[-1][1]
    assert sorted(d for d, _ in seen) == list(range(1, total + 1)), (
        "并行下丢了进度计数")


# --------------------------------------------------------------------------
#  8. MC 内核的原地改写
# --------------------------------------------------------------------------

def _mcgbmq_reference(s0, r, sigma, T, nPath, nStep, seed=20):
    """改写前的写法，逐字保留，作为逐位比对的黄金基准。

    McGbmQ 现在全程在一块缓冲上原地推进（省 4.5 倍峰值内存）。它必须与这个
    参考实现**逐位相同**——随机流、浮点运算顺序都不能变。
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((nPath // 2, nStep))
    W = np.r_[W1, -W1]
    h = T / nStep
    dlogS = (r - 0.5 * sigma ** 2) * h + sigma * np.sqrt(h) * W
    return s0 * np.exp(np.cumsum(dlogS, 1))


@pytest.mark.parametrize("nPath,nStep", [
    (2, 1), (10, 3), (1000, 20), (2000, 61), (4000, 243),
])
@pytest.mark.parametrize("seed", [20, 0, 12345])
def test_mcgbmq_inplace_matches_reference(nPath, nStep, seed):
    from pricing.mc_engine import McGbmQ

    args = (100.0, 0.03, 0.2, nStep / 243, nPath, nStep)
    assert np.array_equal(
        _mcgbmq_reference(*args, seed=seed), McGbmQ(*args, seed=seed)), (
        "原地改写改变了数值")


def test_mcgbmq_keeps_antithetic_structure():
    """对偶变量：上下两半的对数收益必须严格相反。"""
    from pricing.mc_engine import McGbmQ

    nPath, nStep = 1000, 12
    s = McGbmQ(100.0, 0.0, 0.2, nStep / 243, nPath, nStep, seed=7)
    half = nPath // 2
    up = np.diff(np.log(s[:half]), axis=1, prepend=np.log(100.0))
    dn = np.diff(np.log(s[half:]), axis=1, prepend=np.log(100.0))
    # 两半的漂移项相同、随机项相反 → 和恒为 2*drift
    assert np.allclose(up + dn, (up + dn)[0, 0], atol=1e-12)


def test_mcgbmq_still_validates_inputs():
    from pricing.mc_engine import McGbmQ

    with pytest.raises(ValueError, match="even"):
        McGbmQ(100.0, 0.03, 0.2, 0.1, nPath=999, nStep=5)
    with pytest.raises(ValueError, match="nStep"):
        McGbmQ(100.0, 0.03, 0.2, 0.1, nPath=100, nStep=0)


def test_mcgbmq_seed_none_still_independent():
    """原地缓冲不能把 seed=None 的独立采样语义弄丢。"""
    from pricing.mc_engine import McGbmQ

    a = McGbmQ(100.0, 0.03, 0.2, 0.1, 1000, 10, seed=None)
    b = McGbmQ(100.0, 0.03, 0.2, 0.1, 1000, 10, seed=None)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
#  9. 累计期权 payoff 的两处修正
# --------------------------------------------------------------------------

_DE_SUBTYPES = (
    "Opt_Decumulator", "Opt_Decumulator_Back", "Opt_Decumulator_Fix",
    "Opt_Decumulator_Fix_E", "Opt_EnDecumulator", "Opt_EnDecumulator_Fix",
    "Opt_ASGQ_call_put", "Opt_ASGQ_EP", "Opt_ASGQ_EF", "Opt_ASGQ_EFF",
    "Opt_ASGQ_DP", "Opt_ASGQ_DF", "Opt_ASGQ_DFF",
)


def _de(subtype, cp=1, npath=2000, t_days=20, t_over=0, sr=()):
    return Option_DE(subtype, 100.0, list(sr), 90.0, t_over, t_days,
                     list(range(1, t_days + t_over + 1)), 0.18, 110.0, 2, cp,
                     nPath=npath, r=0.03, q=0.03,
                     fix=2.0, P=95.0, amount=1.5)


@pytest.mark.parametrize("subtype", _DE_SUBTYPES)
@pytest.mark.parametrize("cp", [1, -1])
def test_decumulator_prices_in_both_directions(subtype, cp):
    """13 个子类型 × 看涨/看跌都必须能定价。

    此前 Opt_ASGQ_EP 与 Opt_ASGQ_EF 的看跌分支漏了 .reshape(nPath, 1)，
    直接抛 numpy 广播错误——而 GUI 的方向选项就摆着「看跌 (Put)」，
    选中即崩。看涨分支一直是对的，修法就是照抄它。
    """
    assert np.isfinite(_de(subtype, cp=cp).get_price())


@pytest.mark.parametrize("subtype", _DE_SUBTYPES)
def test_decumulator_discount_factor_broadcasts(subtype):
    """贴现因子只沿观察日变化，必须是一维的。

    它原本被 np.tile 成 nPath × le 的矩阵，而每一行都一模一样——
    nPath=1e5、le=243 时白占 194 MB。改成广播后数值逐位不变
    （见 test_decumulator_matches_tiled_discount_reference）。
    """
    import inspect

    source = inspect.getsource(getattr(Option_DE, subtype))
    if "discount_factor" not in source:
        pytest.skip("该子类型不使用贴现因子")
    assert "np.tile(np.exp(" not in source, "贴现因子又被物化成矩阵了"




# 黄金值。贴现因子广播化 + 看跌分支补 reshape 都不该
# 动到任何一个已经能算出来的价格。EP / EF 的看跌组合不在表里——它们
# 改动前直接抛广播错误，根本没有"原值"可对。
_DE_GOLDEN = {
    "Opt_ASGQ_DFF|-1|0|0": 29.996296524910573,
    "Opt_ASGQ_DFF|-1|5|5": 37.5,
    "Opt_ASGQ_DFF|1|0|0": 39.276396706790784,
    "Opt_ASGQ_DFF|1|5|5": 49.276396706790756,
    "Opt_ASGQ_DF|-1|0|0": 29.996296524910573,
    "Opt_ASGQ_DF|-1|5|5": 37.5,
    "Opt_ASGQ_DF|1|0|0": 194.09630859972813,
    "Opt_ASGQ_DF|1|5|5": 249.09630859972813,
    "Opt_ASGQ_DP|-1|0|0": -99.9903979595688,
    "Opt_ASGQ_DP|-1|5|5": -125.0,
    "Opt_ASGQ_DP|1|0|0": 198.20825019793935,
    "Opt_ASGQ_DP|1|5|5": 253.20825019793935,
    "Opt_ASGQ_EFF|-1|0|0": 29.996296524910573,
    "Opt_ASGQ_EFF|-1|5|5": 37.5,
    "Opt_ASGQ_EFF|1|0|0": 38.874914662386686,
    "Opt_ASGQ_EFF|1|5|5": 48.73798099675001,
    "Opt_ASGQ_EF|-1|5|5": 37.5,
    "Opt_ASGQ_EF|1|0|0": 193.69482655532406,
    "Opt_ASGQ_EF|1|5|5": 248.5578928896874,
    "Opt_ASGQ_EP|-1|5|5": -125.0,
    "Opt_ASGQ_EP|1|0|0": 197.80676815353524,
    "Opt_ASGQ_EP|1|5|5": 252.66983448789856,
    "Opt_ASGQ_call_put|-1|0|0": -99.9903979595688,
    "Opt_ASGQ_call_put|-1|5|5": -125.0,
    "Opt_ASGQ_call_put|1|0|0": 197.72752246535975,
    "Opt_ASGQ_call_put|1|5|5": 247.52191628001697,
    "Opt_Decumulator_Back|-1|0|0": -399.8348821130444,
    "Opt_Decumulator_Back|-1|5|5": -509.8348821130444,
    "Opt_Decumulator_Back|1|0|0": 195.38495290788086,
    "Opt_Decumulator_Back|1|5|5": 250.38495290788086,
    "Opt_Decumulator_Fix_E|-1|0|0": -400.21587267167934,
    "Opt_Decumulator_Fix_E|-1|5|5": -500.2698408395993,
    "Opt_Decumulator_Fix_E|1|0|0": 38.22602029428241,
    "Opt_Decumulator_Fix_E|1|5|5": 47.952152963009084,
    "Opt_Decumulator_Fix|-1|0|0": -399.8348821130444,
    "Opt_Decumulator_Fix|-1|5|5": -509.8348821130444,
    "Opt_Decumulator_Fix|1|0|0": 39.0289843830906,
    "Opt_Decumulator_Fix|1|5|5": 49.028984383090574,
    "Opt_Decumulator|-1|0|0": 0.0,
    "Opt_Decumulator|-1|5|5": 0.0,
    "Opt_Decumulator|1|0|0": 193.6612927617474,
    "Opt_Decumulator|1|5|5": 248.6612927617474,
    "Opt_EnDecumulator_Fix|-1|0|0": -199.80607512901065,
    "Opt_EnDecumulator_Fix|-1|5|5": -264.80607512901065,
    "Opt_EnDecumulator_Fix|1|0|0": 39.34706116717105,
    "Opt_EnDecumulator_Fix|1|5|5": 49.34706116717102,
    "Opt_EnDecumulator|-1|0|0": -199.80607512901065,
    "Opt_EnDecumulator|-1|5|5": -264.80607512901065,
    "Opt_EnDecumulator|1|0|0": 195.7030296919613,
    "Opt_EnDecumulator|1|5|5": 250.7030296919613,
}


@pytest.mark.parametrize("key", sorted(_DE_GOLDEN))
def test_decumulator_prices_unchanged_by_payoff_cleanup(key):
    """价格必须与基准值相符，容差 1e-12 相对。

    **这条不能用精确相等。** 黄金值是在一台机器上生成的常量，而 numpy 的
    ``exp`` / ``cumsum`` 在不同平台走不同的 SIMD 与 libm，末位会差 1~2 ULP
    （实测 macOS arm64 与 ubuntu x86_64 之间相对差 1.4e-16 ~ 2.9e-16）。
    最初写成 ``==`` 的版本在 CI 上必红——本机全绿，推上去才发现。

    1e-12 的取法：比平台噪声（~3e-16）高 4 个数量级，比这套测试要拦的最小
    一次真实改动（熔断腿贴现口径，2e-4）低 8 个数量级。中间留着足够宽的
    判别带，两头都不会误判。

    同一进程内两种写法的对拍（``assert_results_identical`` 那一组）仍然用
    精确相等——那里没有跨平台问题，一个 ULP 的差就是真的有人动了数值。
    """
    subtype, cp, t_over, nsr = key.split("|")
    sr = [100.0 + i * 0.5 for i in range(int(nsr))]
    price = _de(subtype, cp=int(cp), t_over=int(t_over), sr=sr).get_price()
    assert float(price) == pytest.approx(
        _DE_GOLDEN[key], rel=1e-12, abs=1e-9), f"{key} 的价格变了"


# --------------------------------------------------------------------------
#  10. 累计期权三个赔付类参数的命名与默认值
# --------------------------------------------------------------------------

def test_decumulator_param_labels_are_distinguishable():
    """fix 与 amount 是两笔不同的钱，界面上不能用同一个词。

    它们此前叫「固定赔付」和「固定金额」，代码注释里更是两个都写成
    「固定赔付」——读的人分不出说的是哪一个。现在：
      fix    = 区间赔付（标的在 K~H 区间时每日结算）
      amount = 熔断赔付（熔断日起每日结算）
    """
    from deltalab_ui.constants import OPTION_CLASSES

    params = dict(
        (name, label)
        for name, label, *_ in OPTION_CLASSES["累计期权 (Decumulator)"]["params"]
    )
    assert params["fix"].startswith("区间赔付")
    assert params["amount"].startswith("熔断赔付")
    assert params["P"].startswith("保障价格")

    labels = {Option_DE._PARAM_LABELS[k] for k in ("fix", "P", "amount")}
    assert len(labels) == 3, "三个参数的中文名撞车了"
    # 谁也不能是谁的前缀——「固定赔付」/「固定金额」当年就是这么混起来的。
    for a in labels:
        for b in labels:
            if a is not b:
                assert not a.startswith(b), f"{a!r} 与 {b!r} 仍不好区分"


def test_protection_price_defaults_to_entry_price():
    """保障价格默认取入场价，让三个 *P 结构开箱可用。

    它此前默认 0，而 0 会被建构闭包当成「未填写」→ 构造期报错。
    """
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    defaults = {name: default for name, _label, _t, default, *_ in cfg["params"]}
    assert defaults["P"] == defaults["s0"] == 100.0

    params = dict(defaults)
    for subtype in ("Opt_ASGQ_call_put", "Opt_ASGQ_EP", "Opt_ASGQ_DP"):
        for cp in (1, -1):
            params["cp"] = cp
            option = cfg["build"](subtype, params)
            option.nPath = 400
            assert np.isfinite(option.get_price()), (
                f"{subtype} cp={cp} 在默认参数下仍不能定价")


def test_structure_docs_do_not_conflate_fix_and_amount():
    """结构说明里也不能再把两笔赔付都写成同一个词。"""
    from deltalab_ui import structure_docs

    text = "\n".join(
        str(v) for v in vars(structure_docs).values() if isinstance(v, dict)
        for v in v.values()
    )
    assert "固定金额" not in text, "结构说明里仍有含糊的「固定金额」"
    assert "双固定赔付" not in text, "结构说明里仍有含糊的「双固定赔付」"


def test_decumulator_rejects_mismatched_observation_days():
    """已实现天数 + 剩余期限必须正好铺满观察日。

    GUI 的建构闭包把 sr 硬编码成 []，observ 却按 T_days + T_over 生成——
    「已过天数」一填非 0，13 个子类型全部在定价深处炸 IndexError，
    报错只给两个数字，看不出是哪个字段的问题。
    """
    # sr 空、已过天数 3：先命中"长度必须等于已过天数"这条更具体的
    with pytest.raises(ValueError, match="必须等于已过天数"):
        Option_DE("Opt_Decumulator", 100.0, [], 90.0, 3, 20,
                  list(range(1, 24)), 0.18, 110.0, 2, 1, nPath=200)
    # sr 长度对但观察日不够：命中总数那条
    with pytest.raises(ValueError, match="观察日数量对不上"):
        Option_DE("Opt_Decumulator", 100.0, [100.0] * 3, 90.0, 3, 20,
                  list(range(1, 20)), 0.18, 110.0, 2, 1, nPath=200)
    # 补上对应的已实现价格就合法了
    Option_DE("Opt_Decumulator", 100.0, [100.0] * 3, 90.0, 3, 20,
              list(range(1, 24)), 0.18, 110.0, 2, 1, nPath=200).get_price()


@pytest.mark.parametrize("cp", [1, -1])
def test_single_day_option_is_not_inflated_by_npath(cp):
    """le == 1 时形状 (nPath,) 与 (nPath, 1) 会广播成 (nPath, nPath)。

    改动前这条路径**不报错**，价格直接放大 nPath 倍——GUI 默认
    nPath=100000，也就是错 10 万倍，同时还要申请一块 80 GB 的数组。
    价格必须与 nPath 无关（只在 MC 噪声范围内变动）。
    """
    prices = [
        _de("Opt_ASGQ_EP", cp=cp, npath=n, t_days=1).get_price()
        for n in (200, 2000, 20000)
    ]
    spread = max(prices) - min(prices)
    scale = max(1.0, max(abs(p) for p in prices))
    assert spread / scale < 0.5, f"价格随 nPath 漂移过大: {prices}"


# --------------------------------------------------------------------------
#  11. 累计期权 13 个结构的中文名
# --------------------------------------------------------------------------
#
# 名字按三个正交维度拼，顺序固定，取默认值的那一段省略不写：
#   ① 触碰障碍 H 之后怎么办（必写）
#   ② 区间 (K, H) 怎么赔（线性省略，按 fix 结算写「区间固赔」）
#   ③ 杠杆腿 (S ≤ K) 怎么观察
# 词表定在这里而不是从 constants import：这几条测试要拦的正是「有人顺手加了
# 第四个说法」，跟着源码走的词表拦不住它。

_BARRIER_WORDS = frozenset({
    "敲出终止", "敲出计零", "敲出增强", "熔断保障", "熔断赔付"})
_BAND_WORDS = frozenset({"区间固赔"})
_LEVERAGE_WORDS = frozenset({"每日杠杆", "到期杠杆", "到期结算"})


def _decumulator_display_names():
    from deltalab_ui.constants import OPTION_CLASSES, SUBTYPE_DISPLAY

    return {
        subtype: SUBTYPE_DISPLAY[subtype]
        for subtype in OPTION_CLASSES["累计期权 (Decumulator)"]["subtypes"]
    }


@pytest.mark.parametrize("subtype", sorted(_decumulator_display_names()))
def test_decumulator_name_decomposes_into_the_declared_dimensions(subtype):
    """每个名字都必须解析成 ①[·②][·③]累计，且分词全部出自受控词表。

    这是「乱」的根因：此前的名字是历史叫法的堆叠，修饰语想放哪放哪——
    「固定赔付回归累计」前置、「固赔到期结算累计」中置、「熔断每日保障累计」
    夹在中间，排序后同族的结构根本排不到一起。
    """
    name = _decumulator_display_names()[subtype]
    assert name.endswith("累计"), name

    tokens = name[:-len("累计")].split("·")
    assert tokens[0] in _BARRIER_WORDS, f"{name}: ① 用了词表外的说法"

    rest = tokens[1:]
    assert len(rest) <= 2, f"{name}: 维度多于三个"
    if len(rest) == 2:
        assert rest[0] in _BAND_WORDS and rest[1] in _LEVERAGE_WORDS, (
            f"{name}: ② 必须排在 ③ 前面")
    elif rest:
        assert rest[0] in _BAND_WORDS | _LEVERAGE_WORDS, (
            f"{name}: ②/③ 用了词表外的说法")


def test_the_leverage_dimension_marks_exactly_the_e_and_d_variants():
    """③ 必须与 ``ASGQ_E*`` / ``ASGQ_D*`` 的真实语义对上。

    这是改名前**唯一真错**的一条：``E`` 与 ``D`` 曾被写成「到期观察熔断」与
    「每日观察熔断」，但七个 ASGQ 的熔断判定都是同一行
    ``np.cumsum(ss >= H, axis=1) > 0``——逐日路径依赖、完全一致。差别只在杠杆
    腿：E 只在到期日看 S_T ≤ K，D 逐日看 S ≤ K。所以那两个名字修饰错了对象，
    而且「熔断每日保障累计」(EP) 与「每日熔断保障累计」(DP) 只差两字语序。
    """
    names = _decumulator_display_names()

    for subtype, name in names.items():
        if subtype.startswith("Opt_ASGQ_E") and subtype != "Opt_ASGQ_EFF":
            assert "到期杠杆" in name, f"{subtype} 是到期观察杠杆腿"
        if subtype.startswith("Opt_ASGQ_D"):
            assert "每日杠杆" in name, f"{subtype} 是每日观察杠杆腿"
        # 熔断观察频率不是维度，任何名字都不该再声称它是。
        assert "熔断每日" not in name and "每日熔断" not in name, name

    assert "到期杠杆" in names["Opt_ASGQ_EFF"]
    # 主项结算频率只区分出这一个结构，用「到期结算」而不是「到期杠杆」。
    assert names["Opt_ASGQ_call_put"] == "熔断保障·到期结算累计"


def test_leveraged_pairs_differ_only_in_the_leverage_segment():
    """D/E 成对的三组，除 ③ 之外必须逐字相同——否则名字看不出它们是一对。"""
    names = _decumulator_display_names()

    for d_key, e_key in (("Opt_ASGQ_DP", "Opt_ASGQ_EP"),
                         ("Opt_ASGQ_DF", "Opt_ASGQ_EF"),
                         ("Opt_ASGQ_DFF", "Opt_ASGQ_EFF")):
        assert (names[d_key].replace("每日杠杆", "〇")
                == names[e_key].replace("到期杠杆", "〇")), (
            f"{names[d_key]} 与 {names[e_key]} 的差别不止杠杆腿")


def test_the_word_gupei_names_only_the_fix_payment():
    """「固赔」只许指 `fix`（区间赔付），不许再兼指 `amount`。

    它曾在同一个下拉里指三样东西：``Decumulator_Fix`` 的固赔是 fix、
    ``ASGQ_EF`` 的固赔是 amount、``*FF`` 的「双固赔」是两个都有。
    """
    from deltalab_ui.constants import OPTION_CLASSES

    names = _decumulator_display_names()
    needs_fix = {
        subtype
        for subtype in OPTION_CLASSES["累计期权 (Decumulator)"]["subtypes"]
        if "fix" in Option_DE._REQUIRED_PARAMS.get(subtype, ())
    }

    for subtype, name in names.items():
        assert ("区间固赔" in name) == (subtype in needs_fix), (
            f"{subtype} 的「区间固赔」与是否真需要 fix 对不上：{name}")
        assert "双固赔" not in name, name


def test_no_display_name_is_a_prefix_of_another():
    """任何一个显示名都不能是另一个的前缀。

    下拉是等宽截断的，前缀关系意味着窄列下两条会显示成同一串字符。
    「固定赔付」/「固定金额」当年就是这么混起来的。
    """
    from deltalab_ui.constants import SUBTYPE_DISPLAY

    names = sorted(SUBTYPE_DISPLAY.values())
    assert len(set(names)) == len(names), "有重名"
    for a in names:
        for b in names:
            if a != b:
                assert not b.startswith(a), f"{a!r} 是 {b!r} 的前缀"


def test_structure_doc_titles_are_generated_from_the_display_names():
    """说明页的标题行必须与显示名逐字相同。

    此前它是手抄的第二份副本，13 条累计结构已经漂开 5 条——``ASGQ_call_put``
    显示名写「到期熔断保障累计」而说明页写「到期观察熔断保障累计」。现在标题
    由 ``SUBTYPE_DISPLAY`` 生成，这条测试保证没人再把它抄回正文里。
    """
    from deltalab_ui.constants import OPTION_CLASSES, SUBTYPE_DISPLAY
    from deltalab_ui.structure_docs import STRUCTURE_DOCS

    expected_keys = {
        (cls_name, subtype)
        for cls_name, cfg in OPTION_CLASSES.items()
        for subtype in cfg["subtypes"]
    }
    assert set(STRUCTURE_DOCS) == expected_keys, "说明与注册表对不上"

    for (_cls_name, subtype), text in STRUCTURE_DOCS.items():
        head = text.splitlines()[0]
        assert head == f"【{SUBTYPE_DISPLAY[subtype]} {subtype}】", head


def test_old_display_names_still_resolve_to_their_internal_keys():
    """旧名走反向映射仍能还原；撞车时赢的必须是新名。

    落盘的数据存的是内部键，所以这张别名表不参与迁移——它兜的是输入侧：
    手输、脚本、从旧文档粘过来的名字。
    """
    from deltalab_ui.constants import (
        SUBTYPE_DISPLAY, SUBTYPE_FROM_DISPLAY, _LEGACY_SUBTYPE_DISPLAY,
    )

    for subtype, legacy in _LEGACY_SUBTYPE_DISPLAY.items():
        assert SUBTYPE_FROM_DISPLAY[legacy] == subtype
    for subtype, current in SUBTYPE_DISPLAY.items():
        assert SUBTYPE_FROM_DISPLAY[current] == subtype


# --------------------------------------------------------------------------
#  11. 障碍档位必须随方向摆到标的价的另一侧
# --------------------------------------------------------------------------

_BARRIER_CLASSES = [
    ("累计期权 (Decumulator)", 1, -1),      # (类名, 正常方向, 反方向)
    ("气囊期权 (Airbag)", 1, -1),
    ("雪球期权 (Snowball)", -1, 1),          # 雪球默认 cp=-1
]


def _class_defaults(cls_name):
    from deltalab_ui.constants import OPTION_CLASSES

    return {name: default
            for name, _label, _dtype, default, *_ in
            OPTION_CLASSES[cls_name]["params"]}


@pytest.mark.parametrize("cls_name,forward,reverse", _BARRIER_CLASSES)
def test_default_direction_passes_validation(cls_name, forward, reverse):
    from deltalab_ui.panel_form import FormPanelMixin

    params = _class_defaults(cls_name)
    assert params["cp"] == forward
    FormPanelMixin._validate_barrier_direction(cls_name, params)


@pytest.mark.parametrize("cls_name,forward,reverse", _BARRIER_CLASSES)
def test_flipping_direction_without_moving_barriers_is_rejected(
        cls_name, forward, reverse):
    """只改方向不动障碍 = 第一天就触发，实测触发概率 1.0000。

    此前这种配置照常算出一串数字，看着正常，其实全是"首日敲掉"的退化值。
    """
    from deltalab_ui.panel_form import FormPanelMixin

    params = _class_defaults(cls_name)
    params["cp"] = reverse
    with pytest.raises(ValueError, match="第一天就会触发"):
        FormPanelMixin._validate_barrier_direction(cls_name, params)


@pytest.mark.parametrize("cls_name,forward,reverse", _BARRIER_CLASSES)
def test_mirrored_barriers_pass_in_the_reverse_direction(
        cls_name, forward, reverse):
    """把档位绕 s0 镜像到另一侧之后，反方向就合法了。"""
    from deltalab_ui.constants import DIRECTIONAL_LEVELS
    from deltalab_ui.panel_form import FormPanelMixin

    params = _class_defaults(cls_name)
    s0 = float(params["s0"])
    for field in DIRECTIONAL_LEVELS[cls_name]["mirror"]:
        params[field] = 2.0 * s0 - float(params[field])
    params["cp"] = reverse
    FormPanelMixin._validate_barrier_direction(cls_name, params)


def test_validation_skips_classes_without_barriers():
    """香草与亚式没有障碍，不该被这条校验波及。"""
    from deltalab_ui.panel_form import FormPanelMixin

    for cls_name in ("香草期权 (Vanilla)", "亚式期权 (Asian)"):
        for cp in (1, -1):
            params = _class_defaults(cls_name)
            params["cp"] = cp
            FormPanelMixin._validate_barrier_direction(cls_name, params)


@pytest.mark.parametrize("cls_name,forward,reverse", _BARRIER_CLASSES)
def test_direction_change_mirrors_untouched_defaults(
        cls_name, forward, reverse):
    """切换方向时，仍是默认值的档位自动镜像；改过的一律不碰。"""
    from types import SimpleNamespace

    from deltalab_ui.constants import DIRECTIONAL_LEVELS
    from deltalab_ui.panel_form import FormPanelMixin

    class _Var:
        def __init__(self, v): self._v = str(v)
        def get(self): return self._v
        def set(self, v): self._v = str(v)

    defaults = _class_defaults(cls_name)
    fields = DIRECTIONAL_LEVELS[cls_name]["mirror"]
    s0 = float(defaults["s0"])

    def _fake(overrides=None):
        entries = {f: (_Var((overrides or {}).get(f, defaults[f])), float, None)
                   for f in fields}
        return SimpleNamespace(
            _class_var=SimpleNamespace(get=lambda: cls_name),
            _param_entries=entries), entries

    # 未改动 → 镜像
    fake, entries = _fake()
    FormPanelMixin._mirror_levels_on_direction_change(fake)
    for f in fields:
        assert float(entries[f][0].get()) == pytest.approx(
            2.0 * s0 - float(defaults[f])), f"{f} 没有被镜像"

    # 再切一次 → 镜像回原值（镜像是对合）
    FormPanelMixin._mirror_levels_on_direction_change(fake)
    for f in fields:
        assert float(entries[f][0].get()) == pytest.approx(float(defaults[f]))

    # 用户改过其中一项 → 整组都不动
    custom = {fields[0]: float(defaults[fields[0]]) + 3.0}
    fake, entries = _fake(custom)
    before = {f: entries[f][0].get() for f in fields}
    FormPanelMixin._mirror_levels_on_direction_change(fake)
    assert {f: entries[f][0].get() for f in fields} == before, (
        "用户改过的档位被静默改写了")


# --------------------------------------------------------------------------
#  12. 熔断当天直接结算
# --------------------------------------------------------------------------

# 七个熔断结构全在内。此前这张表只列了三个「熔断后结算依赖价格」的，
# 另外四个被判为"付常数 amount，不受影响"——那只说对了**金额**，没说**结算日**，
# 于是它们的熔断腿一直逐日铺开贴现，与早退分支矛盾了整整一轮。
_KO_SETTLED_SUBTYPES = ("Opt_ASGQ_EP", "Opt_ASGQ_DP", "Opt_ASGQ_call_put",
                        "Opt_ASGQ_EF", "Opt_ASGQ_DF",
                        "Opt_ASGQ_EFF", "Opt_ASGQ_DFF")


@pytest.mark.parametrize("subtype", _KO_SETTLED_SUBTYPES)
@pytest.mark.parametrize("cp", [1, -1])
def test_knockout_settlement_is_split_invariant(subtype, cp):
    """熔断日落在已实现段还是模拟段，价格必须一样。

    实务口径：熔断当天直接结算——熔断日及之后每天的收益都等于
    「熔断日价格 − 保障价格」这个常数，期权随即终止。

    早退分支一直是这么写的，MC 分支却不是：``EP``/``DP`` 用当天价格、
    ``call_put`` 用到期价格。于是同一条路径，只把熔断日从模拟段挪进已实现段，
    价格能差十倍——而对冲回测每天都往已实现段追加收盘价，标的一穿越障碍，
    当天估值就从一条路径跳到另一条。

    这条把两支钉在一起：给定同一条确定性路径，无论怎么切分都必须同价。

    用 r=0 是有意的：每换一个切分点，"现在"这个估值时点也跟着移动，
    r>0 时贴现距离本来就不同，价格**应该**不一样。r=0 把贴现摘掉，
    剩下的就纯粹是金额口径——那正是这条要钉的东西。
    """
    de_mod = sys.modules["pricing.Option_DE"]

    if cp == 1:
        path = np.array([100., 102., 104., 108., 103., 106., 99., 97.])
        K, H, P = 95.0, 105.0, 100.0
    else:
        path = np.array([100., 98., 94., 90., 95., 92., 101., 103.])
        K, H, P = 105.0, 95.0, 100.0
    total = len(path)

    real_mc = de_mod.McGbmQ

    def _fixed(_s0, _r, _sigma, _T, n_path, n_step, seed=20, **_kw):
        return np.tile(path[total - n_step:], (n_path, 1))

    de_mod.McGbmQ = _fixed
    try:
        prices = []
        for simulated in range(1, total):
            option = de_mod.Option_DE(
                subtype, float(path[total - simulated - 1]),
                list(path[:total - simulated]),
                K, total - simulated, simulated, list(range(1, total + 1)),
                0.18, H, 2, cp, nPath=4, r=0.0, q=0.0,
                P=P, amount=3.0, fix=2.0)
            prices.append(round(float(option.get_price()), 9))
    finally:
        de_mod.McGbmQ = real_mc

    assert len(set(prices)) == 1, (
        f"同一条路径按不同方式切分算出了 {len(set(prices))} 个价格: "
        f"{sorted(set(prices))}")


# 熔断后按「熔断日价格 − 保障价格」结算的（金额依赖价格）
_KO_PRICE_LINKED = ("Opt_ASGQ_EP", "Opt_ASGQ_DP", "Opt_ASGQ_call_put")
# 熔断后按常数「熔断赔付」结算的（金额与价格无关）
_KO_FIXED_AMOUNT = ("Opt_ASGQ_EF", "Opt_ASGQ_DF",
                    "Opt_ASGQ_EFF", "Opt_ASGQ_DFF")


@pytest.mark.parametrize("subtype", _KO_SETTLED_SUBTYPES)
def test_knockout_amount_is_frozen_at_the_barrier_day(subtype):
    """熔断后的每日收益是常数，与熔断之后标的怎么走无关。

    改熔断日**之后**的价格，定价一律不能变。改熔断日**当天**的价格：
    保障价族（金额 = 熔断日价格 − 保障价）必须变；
    熔断赔付族（金额是常数 amount）必须不变——两族的断言方向相反，
    合起来才说明"冻结"这件事真的落到了正确的量上。
    """
    de_mod = sys.modules["pricing.Option_DE"]
    K, H, P = 95.0, 105.0, 100.0
    real_mc = de_mod.McGbmQ

    def _price(path):
        def _fixed(_s0, _r, _sigma, _T, n_path, n_step, seed=20, **_kw):
            return np.tile(path[len(path) - n_step:], (n_path, 1))
        de_mod.McGbmQ = _fixed
        try:
            return float(de_mod.Option_DE(
                subtype, float(path[0]), [], K, 0, len(path),
                list(range(1, len(path) + 1)), 0.18, H, 2, 1,
                nPath=4, r=0.0, q=0.0,
                P=P, amount=3.0, fix=2.0).get_price())
        finally:
            de_mod.McGbmQ = real_mc

    base = np.array([100., 102., 108., 103., 99., 97.])   # 第 2 天熔断
    after = base.copy(); after[3:] = [130., 140., 150.]   # 只动熔断之后
    on_day = base.copy(); on_day[2] = 120.                # 动熔断当天

    assert _price(base) == _price(after), "熔断之后的走势竟然影响了价格"
    if subtype in _KO_PRICE_LINKED:
        assert _price(base) != _price(on_day), "熔断当天的价格没有进入结算"
    else:
        assert _price(base) == _price(on_day), (
            "熔断赔付是常数，不该受熔断当天价格影响")


@pytest.mark.parametrize("subtype", _KO_SETTLED_SUBTYPES)
@pytest.mark.parametrize("r", [0.03, 0.05])
def test_knockout_leg_discounts_from_the_barrier_day(subtype, r):
    """r>0 下的**精确**不变量，比 r=0 的切分不变量强得多。

    取同一条确定性路径，比较「熔断日是模拟段第一天」与「熔断日已进 sr」
    两种摆法。两者的已实现现金流完全相同（``observ <= T_over`` 的贴现因子
    恒为 1），所以价差**必须恰好**等于熔断腿隔一天的贴现：

        逐日结算族   V(j+1) − V(j) = 熔断腿名义 × (1 − e^(−r·dt))
        到期一次结算 V(j+1)        = V(j) × e^(r·dt)

    r=0 时两边都是 0，所以那条测试对"贴现口径漏改"结构性失明——
    ``EF``/``DF``/``EFF``/``DFF`` 就是这么漏过去的。这条不留那个口子。
    """
    de_mod = sys.modules["pricing.Option_DE"]
    path = np.array([100., 101., 103., 106., 112., 108.,
                     104., 102., 99., 97., 95., 93.])
    le, ko_at = len(path), 4                  # 第 4 天穿越 H=110
    K, H, P, amount, fix, N = 100.0, 110.0, 104.0, 3.0, 2.0, 3
    real_mc = de_mod.McGbmQ

    def _price(elapsed):
        def _fixed(_s0, _r, _sigma, _T, n_path, n_step, seed=20, **_kw):
            return np.tile(path[le - n_step:], (n_path, 1))
        de_mod.McGbmQ = _fixed
        try:
            return float(de_mod.Option_DE(
                subtype, float(path[elapsed - 1]), list(path[:elapsed]),
                K, elapsed, le - elapsed, list(range(1, le + 1)),
                0.20, H, N, 1, r=r, q=0.0, nPath=4,
                P=P, amount=amount, fix=fix).get_price())
        finally:
            de_mod.McGbmQ = real_mc

    before, after = _price(ko_at), _price(ko_at + 1)
    one_day = 1.0 - np.exp(-r / 243.0)
    n_ko = le - ko_at
    if subtype == "Opt_ASGQ_call_put":
        expected = before * (np.exp(r / 243.0) - 1.0)
    elif subtype in ("Opt_ASGQ_EP", "Opt_ASGQ_DP"):
        expected = (path[ko_at] - P) * n_ko * one_day
    else:
        expected = amount * n_ko * one_day

    assert after - before == pytest.approx(expected, abs=1e-9), (
        f"熔断腿的贴现日不对：实际差 {after - before:.9f}，"
        f"应有 {expected:.9f}")


def test_cache_key_tracks_the_pricer_version():
    """定价口径一改，旧缓存必须失效。

    key material 的其余各项刻画的都是**输入**（期权属性、行情切片、策略、
    回测参数），没有一项能察觉「定价代码换了口径」。熔断结算口径改过之后，
    盘上的逐 bar 缓存全是旧口径算的，而 ``_digest_object`` 对新旧两版
    Option_DE 算出的摘要**完全相同**——不加这个版本号，旧结果会被当成有效
    命中原样读回，不报错也不告警。
    """
    import history_bar_cache as cache

    assert isinstance(cache.PRICER_VERSION, int)
    assert cache.PRICER_VERSION >= 2, "口径改过就要 +1"

    # **两条 key 都要查**。它们写进同一个目录、共用 store_by_key /
    # load_by_key，少一处版本号，那条路径上的旧口径缓存照样会被当成有效
    # 命中——只查 key_for 的守卫正是这么漏掉 key_for_recipe 的。
    import inspect
    for fn in (cache.key_for, cache.key_for_recipe):
        assert "PRICER_VERSION" in inspect.getsource(fn), (
            f"{fn.__name__} 的 key material 里没有版本号")


def test_pricer_version_actually_invalidates_both_key_kinds(tmp_path):
    """版本号一变，两条 key 都必须变——这是「旧口径缓存失效」的唯一保证。"""
    import history_bar_cache as cache

    from pricing import HistoryReplaySpec

    recipe = {"prices": [100.0, 101.0], "strategy_name": "close_to_close"}
    spec = HistoryReplaySpec(
        lookback="month", window_id="segment_1", option=_airbag(npath=200),
        external_path=np.array([100.0, 101.0, 102.0]),
        evaluation_days=2, steps_per_day=1,
        strategies={"close_to_close": CloseToCloseStrategy()},
        backtest_kwargs={}, warmup_kwargs={}, metadata={})

    before = (cache.key_for(spec, "close_to_close"),
              cache.key_for_recipe(recipe))
    saved = cache.PRICER_VERSION
    cache.PRICER_VERSION = saved + 1
    try:
        after = (cache.key_for(spec, "close_to_close"),
                 cache.key_for_recipe(recipe))
    finally:
        cache.PRICER_VERSION = saved

    assert before[0] != after[0], "key_for 不随定价口径版本变化"
    assert before[1] != after[1], "key_for_recipe 不随定价口径版本变化"


@pytest.mark.parametrize("subtype", _KO_SETTLED_SUBTYPES)
@pytest.mark.parametrize("r", [0.03, 0.05])
def test_no_artificial_jump_across_the_barrier(subtype, r):
    """标的擦着障碍过去与恰好触碰，价差只能是**真实经济差**。

    熔断的只有末日那一天，所以两种情形的差别就该只是那一天收什么：
    保障价族收 (S−P) 而非 (S−K)、熔断赔付族收 amount 而非当日累计。

    此前 ``call_put`` 在这里凭空跳 1.49%：熔断腿改成「熔断日一次结算」之后，
    未熔断腿仍留在「逐日结算」，于是跨越障碍的一瞬间**全部 le 天**的折现
    口径一起翻面，多出 (S_T−K)·(Σdf_j − le·df_T)。现在两侧统一按到期一次
    折现（期权按保证金逐日盯市估值，但现金流当作到期一次结清）。

    这一项在 r=0 时恒为 0，所以必须在 r>0 下测。
    """
    de_mod = sys.modules["pricing.Option_DE"]
    le, K, H, P, amount, fix, N = 243, 95.0, 110.0, 105.0, 3.0, 2.0, 3
    real_mc = de_mod.McGbmQ

    def _price(terminal):
        path = np.full(le, 100.0)
        path[-1] = terminal

        def _fixed(_s0, _r, _sigma, _T, n_path, n_step, seed=20, **_kw):
            return np.tile(path[le - n_step:], (n_path, 1))

        de_mod.McGbmQ = _fixed
        try:
            return float(de_mod.Option_DE(
                subtype, 100.0, [], K, 0, le, list(range(1, le + 1)),
                0.20, H, N, 1, r=r, q=0.0, nPath=4,
                P=P, amount=amount, fix=fix).get_price())
        finally:
            de_mod.McGbmQ = real_mc

    df_expiry = np.exp(-r * le / 243.0)
    if subtype in _KO_PRICE_LINKED:
        economic = (K - P) * df_expiry            # (S−P) 取代 (S−K)
    elif subtype in ("Opt_ASGQ_EF", "Opt_ASGQ_DF"):
        economic = (amount - (H - K)) * df_expiry  # amount 取代 (S−K)
    else:
        economic = (amount - fix) * df_expiry      # amount 取代区间赔付 fix

    jump = _price(H) - _price(H - 1e-9)
    assert jump == pytest.approx(economic, abs=1e-5), (
        f"障碍处有人造跳变 {jump - economic:+.6f}"
        f"（实际 {jump:+.6f}，应有 {economic:+.6f}）")


# --------------------------------------------------------------------------
#  13. 已过天数 × 已实现序列
# --------------------------------------------------------------------------

def test_realized_series_field_is_paired_with_elapsed_days():
    """「已过天数」必须有配对的「已实现序列」输入，否则它是条死路。

    此前建构闭包把已实现序列硬编码成空，而「已过天数」是可编辑字段：
    一填非 0 就必崩（13 个子类型无一幸免），界面上又无处补那几天的收盘价。
    """
    from deltalab_ui.constants import OPTION_CLASSES

    params = OPTION_CLASSES["累计期权 (Decumulator)"]["params"]
    names = [spec[0] for spec in params]
    assert "sr" in names, "缺少已实现序列字段"
    assert names.index("sr") == names.index("T_over") + 1, (
        "已实现序列应当紧跟在已过天数后面")
    sr_spec = next(s for s in params if s[0] == "sr")
    assert sr_spec[2] is list and sr_spec[3] == "", "默认应当是空序列"


@pytest.mark.parametrize("raw,expected", [
    (None, []),
    ("", []),
    ("   ", []),
    ("100", [100.0]),
    ("100,101.5", [100.0, 101.5]),
    ("100 101.5", [100.0, 101.5]),
    ("100，101.5", [100.0, 101.5]),      # 中文逗号
    ("100;101.5", [100.0, 101.5]),
    ([100.0, 101.5], [100.0, 101.5]),   # 快照重放直接传列表
    ((100.0, 101.5), [100.0, 101.5]),
])
def test_realized_series_parsing(raw, expected):
    from deltalab_ui.constants import _parse_number_sequence

    assert _parse_number_sequence(raw, "已实现序列") == expected


def test_realized_series_parsing_accepts_ndarray():
    from deltalab_ui.constants import _parse_number_sequence

    assert _parse_number_sequence(
        np.array([100.0, 101.5]), "已实现序列") == [100.0, 101.5]


@pytest.mark.parametrize("bad", ["abc", "100,abc", "100,-5", "100,0", "100,nan"])
def test_realized_series_rejects_bad_input(bad):
    from deltalab_ui.constants import _parse_number_sequence

    with pytest.raises(ValueError):
        _parse_number_sequence(bad, "已实现序列")


@pytest.mark.parametrize("subtype", _DE_SUBTYPES)
def test_elapsed_days_with_realized_series_prices(subtype):
    """填齐两项之后，13 个子类型都要能定价。"""
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    params = {name: default
              for name, _label, _dtype, default, *_ in cfg["params"]}
    params.update(s0=100.0, T_over=3, sr="100,101,99.5",
                  fix=2.0, P=95.0, amount=1.5)
    option = cfg["build"](subtype, params)
    option.nPath = 400
    assert np.isfinite(option.get_price())


def test_elapsed_days_default_is_unchanged():
    """默认 0 / 空时行为与从前完全一致——这是不回归的底线。"""
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    params = {name: default
              for name, _label, _dtype, default, *_ in cfg["params"]}
    params["s0"] = 100.0
    option = cfg["build"]("Opt_Decumulator", params)
    assert option.sr == []
    assert option.T_over == 0
    assert len(option.observ) == option.T_days


def test_mismatched_realized_series_names_both_fields():
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    params = {name: default
              for name, _label, _dtype, default, *_ in cfg["params"]}
    params.update(s0=100.0, T_over=3, sr="100,101")
    with pytest.raises(ValueError, match="必须等于已过天数"):
        cfg["build"]("Opt_Decumulator", params)


# --------------------------------------------------------------------------
#  14. 构造期的 **kwargs 不再静默吞参数
# --------------------------------------------------------------------------

def _all_option_factories():
    """五个大类各一个可调用的构造器，只差 **kwargs。"""
    from pricing.Option_AS import Option_AS
    from pricing.Option_SNB import Option_SNB
    from pricing.Option_Vanilla import Option_Vanilla

    return {
        "Option_AB": lambda **k: Option_AB(
            "Opt_Airbag", 100.0, [], 100.0, 90.0, 20, list(range(1, 21)),
            0.20, 0.8, 1.0, 1, nPath=400, **k),
        "Option_AS": lambda **k: Option_AS(
            "Asian", 100.0, [], 100.0, 100.0, 22, 22, 0.15, 1,
            0.0, 999999.0, nPath=400, **k),
        "Option_DE": lambda **k: Option_DE(
            "Opt_Decumulator", 100.0, [], 90.0, 0, 20, list(range(1, 21)),
            0.18, 110.0, 2, 1, nPath=400, **k),
        "Option_SNB": lambda **k: Option_SNB(
            "Opt_Snowball", 100.0, 100.0, 100.0, 80.0, 103.0, 40,
            0.18, 0.15, 0.15, 0.2, 1, -1, sr=[], ko_observ=[20, 40],
            nPath=400, margin_call=False, **k),
        "Option_Vanilla": lambda **k: Option_Vanilla(
            "Eu", 100.0, [], 100.0, 20, 0.20, 1, **k),
    }


@pytest.mark.parametrize("name", sorted(_all_option_factories()))
def test_mc_seed_can_be_set_at_construction(name):
    """``mc_seed=`` 必须真的生效。

    它是 OptionBase 的**类属性**、不是各子类 __init__ 的形参，此前会掉进
    ``**kwargs`` 里被静默吞掉——不报错也不生效，所有实例继续共用同一批随机
    数。写"不同 seed 跑多次取标准差"的脚本最容易在这里翻车：标准差恒为 0，
    看着像是定价稳得离谱。
    """
    make = _all_option_factories()[name]
    first, second = make(mc_seed=1), make(mc_seed=2)
    assert (first.mc_seed, second.mc_seed) == (1, 2)
    if name == "Option_Vanilla":
        pytest.skip("解析定价，与随机数无关")
    assert first.get_price() != second.get_price(), (
        "换了 mc_seed 价格却一模一样——种子没有落到实例上")


@pytest.mark.parametrize("name", sorted(_all_option_factories()))
def test_greeks_npath_can_be_set_at_construction(name):
    make = _all_option_factories()[name]
    assert make(greeks_nPath=100).greeks_nPath == 100


@pytest.mark.parametrize("name", sorted(_all_option_factories()))
def test_unknown_keyword_is_rejected(name):
    """拼错的参数名要报错，而不是被 ``**kwargs`` 悄悄丢掉。"""
    make = _all_option_factories()[name]
    with pytest.raises(TypeError, match="无法识别的参数"):
        make(mc_sed=7)          # 少一个 e
    with pytest.raises(TypeError, match="typo_param"):
        make(typo_param=1)


def test_run_multi_still_varies_seed_per_path():
    """run_multi 的 per-path seed 是构造后赋属性，不受这次改动影响。"""
    prices = _path(n=12, seed=3)
    paths = np.tile(prices, (6, 1))
    result = HedgeBacktest(
        _airbag(npath=600, t_days=12), prices,
        strategy=CloseToCloseStrategy(),
    ).run_multi(paths, base_seed=100)
    errors = result["errors"][np.isfinite(result["errors"])]
    assert errors.size >= 2
    assert float(np.std(errors)) > 0, "各路径的 MC 采样没有拉开"


# --------------------------------------------------------------------------
#  15. theta 的 CRN
# --------------------------------------------------------------------------

def test_mcgbmq_draw_steps_defaults_to_bit_identical():
    """不传 draw_steps 时必须与从前逐位相同。"""
    from pricing.mc_engine import McGbmQ

    args = (100.0, 0.03, 0.2, 20 / 243, 1000, 20)
    assert np.array_equal(McGbmQ(*args, seed=20),
                          McGbmQ(*args, seed=20, draw_steps=20))


def test_mcgbmq_draw_steps_shares_the_underlying_normals():
    """按 T 抽、切前 T−1 列，底层标准正态必须与按 T 抽的前 T−1 列逐位相同。

    行优先填充下 ``(n, T)`` 的前 T−1 列 ≠ ``(n, T−1)``，所以 theta 的 bump
    必须显式按 T 抽——这正是 draw_steps 存在的理由。
    """
    from pricing.mc_engine import McGbmQ

    s0, r, sigma, n_path = 100.0, 0.03, 0.2, 2000
    full = McGbmQ(s0, r, sigma, 20 / 243, n_path, 20, seed=20)
    bumped = McGbmQ(s0, r, sigma, 19 / 243, n_path, 19, seed=20, draw_steps=20)
    naive = McGbmQ(s0, r, sigma, 19 / 243, n_path, 19, seed=20)

    def _normals(paths, n_step, horizon):
        h = horizon / n_step
        logs = np.diff(np.log(np.c_[np.full((len(paths), 1), s0), paths]),
                       axis=1)
        return (logs - (r - 0.5 * sigma ** 2) * h) / (sigma * np.sqrt(h))

    shared = _normals(full, 20, 20 / 243)[:, :19]
    assert np.allclose(shared, _normals(bumped, 19, 19 / 243), atol=1e-12)
    assert not np.allclose(shared, _normals(naive, 19, 19 / 243), atol=1e-9)


def test_mcgbmq_rejects_draw_steps_below_nstep():
    from pricing.mc_engine import McGbmQ

    with pytest.raises(ValueError, match="draw_steps"):
        McGbmQ(100.0, 0.03, 0.2, 0.1, 100, 20, draw_steps=19)


def _naive_theta(option):
    """还原改动前的 theta：bump 不锁定抽样步数，两次定价各抽各的随机数。"""
    from pricing.constants import ANNUAL_DAYS

    price0 = option.priced()
    bumped = option._bumped_copy(**option._theta_overrides(1))
    return (bumped.get_price() - price0) / (1 / ANNUAL_DAYS)


@pytest.mark.parametrize("make_option", [_airbag, _snowball, _decumulator])
def test_theta_crn_leaves_other_greeks_untouched(make_option):
    """只有 theta 的 bump 该拿到 mc_draw_steps。

    delta / gamma / vega / rho 的 bump 都不改 nStep，本来就共享随机数。
    把 mc_draw_steps 设到期权本体上（相当于全局施加）会让它们全都变——
    这条正是用来挡住那种改法的。
    """
    baseline = make_option().get_greeks()

    globally_forced = make_option()
    globally_forced.mc_draw_steps = int(globally_forced._time_remaining) + 5
    forced = globally_forced.get_greeks()

    for idx, name in ((0, "delta"), (1, "gamma"), (2, "vega"), (4, "rho")):
        assert baseline[idx] != forced[idx], (
            f"{name} 对 mc_draw_steps 无反应——注入点可能写错了位置")


@pytest.mark.parametrize("make_option", [_airbag, _snowball, _decumulator])
def test_theta_noise_is_reduced_by_crn(make_option):
    """CRN 下的 theta 噪声必须显著小于 bump 自己重抽的版本。

    此前 theta 是两次**相互独立**的 MC 估计之差——实测标准差比修复后大
    1.8~11.6 倍，气囊与雪球连正负号都是错的。
    """
    crn, naive = [], []
    for seed in range(8):
        option = make_option()
        option.mc_seed = seed
        crn.append(option.get_greeks()[3])

        other = make_option()
        other.mc_seed = seed
        naive.append(_naive_theta(other))

    assert np.std(crn, ddof=1) < np.std(naive, ddof=1), (
        f"CRN 没有降低 theta 噪声: {np.std(crn, ddof=1):.4f} "
        f"vs {np.std(naive, ddof=1):.4f}")


# --------------------------------------------------------------------------
#  16. 渲染失败不该丢掉整轮计算
# --------------------------------------------------------------------------

_WINDOWS = {"month": {"segment_1": {"daily": {}}}}


def _rescue_fake(tmp_path, monkeypatch, answer):
    """造一个能跑 _deliver_history_recommendation 的假 self。"""
    from types import SimpleNamespace

    import history_store
    from deltalab_ui import runner as runner_mod

    monkeypatch.setattr(history_store, "results_dir", lambda: str(tmp_path))
    monkeypatch.setattr(runner_mod.messagebox, "showerror",
                        lambda *a, **k: None)
    monkeypatch.setattr(runner_mod.messagebox, "showinfo",
                        lambda *a, **k: None)
    monkeypatch.setattr(runner_mod.messagebox, "askyesno",
                        lambda *a, **k: answer)

    ranking = pd.DataFrame([{
        "lookback": "month", "strategy": "daily", "complete_window": True,
        "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
    }])
    finished = []

    def _boom(*_a, **_k):
        raise RuntimeError("渲染炸了")

    fake = SimpleNamespace(
        _history_pending_result={
            "recommendations": pd.DataFrame(),
            "ranking": ranking,
            "notes": None,
            "window_results": _WINDOWS,
            "source_label": "CSV · x.csv · close",
            "history_state": {"source": "csv", "csv_path": "x.csv"},
            "objective": "incremental_pnl",
        },
        _show_history_recommendation=_boom,
        _finish_history_recommendation=finished.append,
    )
    return fake, finished, ranking


def test_render_failure_keeps_the_result_and_offers_to_save(
        tmp_path, monkeypatch):
    """渲染出错时结果不能丢：用户同意就落盘，之后可以载入。

    此前结果只活在 ``after`` 那个 lambda 的闭包里，渲染一抛异常，十几分钟
    的计算连同 gzip 后仅一百多 KB 的结果包一起没了，只能重跑。
    """
    import history_store
    from deltalab_ui.runner import RunnerMixin

    fake, finished, ranking = _rescue_fake(tmp_path, monkeypatch, answer=True)
    RunnerMixin._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, None, _WINDOWS,
        "CSV · x.csv · close", {"source": "csv"})

    assert finished == [False], "失败应当如实收尾"
    saved = history_store.list_results()
    assert len(saved) == 1, "同意抢救后没有落盘"
    assert saved[0]["label"] == "渲染失败抢救"
    assert fake._history_pending_result is None, "抢救后没有清掉暂存"


def test_render_failure_respects_a_declined_rescue(tmp_path, monkeypatch):
    """用户拒绝就不写盘，但暂存同样要清掉，免得下一轮读到旧结果。"""
    import history_store
    from deltalab_ui.runner import RunnerMixin

    fake, finished, ranking = _rescue_fake(tmp_path, monkeypatch, answer=False)
    RunnerMixin._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, None, _WINDOWS,
        "CSV · x.csv · close", {"source": "csv"})

    assert finished == [False]
    assert history_store.list_results() == []
    assert fake._history_pending_result is None


def test_successful_render_clears_the_pending_result(tmp_path, monkeypatch):
    """渲染成功后暂存必须清掉——留着会让下一轮的失败抢救出旧数据。"""
    from types import SimpleNamespace

    from deltalab_ui.runner import RunnerMixin

    finished = []
    fake = SimpleNamespace(
        _history_pending_result={"ranking": "旧的"},
        _show_history_recommendation=lambda *a, **k: None,
        _finish_history_recommendation=finished.append,
        _latest_history_state=None,
        _latest_history_source_label=None,
    )
    ranking = pd.DataFrame([{
        "lookback": "month", "strategy": "daily", "complete_window": True,
        "rolling_windows": 1, "daily_net_pnl_rms": 2.0,
    }])
    RunnerMixin._deliver_history_recommendation(
        fake, pd.DataFrame(), ranking, None, _WINDOWS, "label",
        {"source": "csv"})

    assert finished == [True]
    assert fake._history_pending_result is None


# --------------------------------------------------------------------------
#  17. 分周期增量出结果
# --------------------------------------------------------------------------

def _period_setup(n=160):
    rng = np.random.default_rng(5)
    history = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
        index=pd.bdate_range("2023-06-01", periods=n), name="close")
    cases = [StrategyCase("c2c", CloseToCloseStrategy(), {})] + [
        StrategyCase(f"b{k}", SigmaBandStrategy(k=k), {})
        for k in (0.5, 1.0, 1.5)]
    return history, cases


def test_period_callback_fires_cheapest_first():
    """回调顺序必须是从便宜到贵——这正是「早出结论」的价值来源。"""
    history, cases = _period_setup()
    seen = []
    recommend_by_rolling_history(
        _airbag(npath=400), history, cases, dict(_SELECTION_KWARGS),
        lookbacks={"week": 5, "month": 20, "quarter": 61}, steps_per_day=1,
        period_callback=lambda label, ranking: seen.append(label))
    assert seen == ["week", "month", "quarter"]


def test_period_ranking_is_already_final():
    """每档回调给出的名次必须与最终全量排名的该档切片逐位相同。

    这是「分周期增量出结果」的全部前提：名次按 lookback 分组算
    （``_rank_rows`` 里的 ``groupby("lookback").cumcount()``），周期之间零
    耦合，所以一档跑完它的名次就已成定局，可以立刻交给用户。
    """
    history, cases = _period_setup()
    partial = {}
    _recs, full, _windows = recommend_by_rolling_history(
        _airbag(npath=400), history, cases, dict(_SELECTION_KWARGS),
        lookbacks={"week": 5, "month": 20, "quarter": 61}, steps_per_day=1,
        period_callback=lambda label, ranking: partial.__setitem__(
            label, ranking))

    assert set(partial) == {"week", "month", "quarter"}
    for label, ranking in partial.items():
        expected = full[full["lookback"] == label].reset_index(drop=True)
        got = ranking.reset_index(drop=True)
        assert list(got.columns) == list(expected.columns)
        numeric = expected.select_dtypes(include=[np.number]).columns
        assert np.array_equal(got[numeric].to_numpy(),
                              expected[numeric].to_numpy(), equal_nan=True), (
            f"{label} 的增量名次与最终不一致")
        other = [c for c in expected.columns if c not in numeric]
        assert got[other].equals(expected[other])


def test_period_callback_failure_does_not_break_the_run():
    """显示用的回调炸了不该把整轮优选带下水。"""
    history, cases = _period_setup()

    def boom(*_args):
        raise RuntimeError("回调炸了")

    _recs, ranking, _w = recommend_by_rolling_history(
        _airbag(npath=400), history, cases, dict(_SELECTION_KWARGS),
        lookbacks={"week": 5, "month": 20}, steps_per_day=1,
        period_callback=boom)
    assert not ranking.empty


def test_selection_without_period_callback_is_unchanged():
    history, cases = _period_setup()
    kwargs = dict(_SELECTION_KWARGS)
    lookbacks = {"week": 5, "month": 20}
    with_cb = recommend_by_rolling_history(
        _airbag(npath=400), history, cases, dict(kwargs),
        lookbacks=lookbacks, steps_per_day=1,
        period_callback=lambda *_a: None)[1]
    without = recommend_by_rolling_history(
        _airbag(npath=400), history, cases, dict(kwargs),
        lookbacks=lookbacks, steps_per_day=1)[1]
    numeric = without.select_dtypes(include=[np.number]).columns
    assert np.array_equal(with_cb[numeric].to_numpy(),
                          without[numeric].to_numpy(), equal_nan=True)


def test_leading_candidate_is_read_defensively():
    from deltalab_ui.runner import RunnerMixin

    assert RunnerMixin._leading_candidate(None) is None
    assert RunnerMixin._leading_candidate(pd.DataFrame()) is None
    assert RunnerMixin._leading_candidate(
        pd.DataFrame([{"nope": 1}])) is None
    table = pd.DataFrame([{"strategy": "b1.0", "rank": 2},
                          {"strategy": "c2c", "rank": 1}])
    assert RunnerMixin._leading_candidate(table) == "c2c"


# --------------------------------------------------------------------------
#  18. 运行期日志
# --------------------------------------------------------------------------

@pytest.fixture
def _fresh_log(tmp_path, monkeypatch):
    """把日志目录指到临时目录，并复位模块状态。"""
    import deltalab_log

    monkeypatch.setattr(deltalab_log, "log_dir",
                        lambda: str(tmp_path / "logs"))
    monkeypatch.setattr(deltalab_log, "log_path",
                        lambda: str(tmp_path / "logs" / "deltalab.log"))
    logger = deltalab_log.get_logger()
    saved = list(logger.handlers)
    logger.handlers.clear()
    monkeypatch.setattr(deltalab_log, "_configured", False)
    yield deltalab_log
    logger.handlers.clear()
    logger.handlers.extend(saved)


def test_logging_writes_to_a_file(_fresh_log):
    """冻结包是 console=False，不落盘就等于什么都没记。"""
    path = _fresh_log.setup(to_stderr=False)
    assert path is not None
    _fresh_log.get_logger("probe").info("启动 frozen=%s", False)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "启动 frozen=False" in text
    assert "deltalab.probe" in text


def test_logging_records_tracebacks(_fresh_log):
    """异常现场要留全，这正是此前 print 到 None 丢掉的东西。"""
    path = _fresh_log.setup(to_stderr=False)
    try:
        raise RuntimeError("模拟渲染失败")
    except RuntimeError:
        _fresh_log.get_logger("probe").exception("渲染失败")
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "Traceback" in text and "模拟渲染失败" in text


def test_logging_setup_is_idempotent(_fresh_log):
    """GUI 入口与测试都可能各调一次，处理器不能叠加。"""
    _fresh_log.setup(to_stderr=False)
    first = len(_fresh_log.get_logger().handlers)
    _fresh_log.setup(to_stderr=False)
    assert len(_fresh_log.get_logger().handlers) == first


def test_logging_degrades_quietly_when_undeletable(_fresh_log, monkeypatch):
    """目录不可写只让日志退化成什么都不记，绝不能把主流程打断。"""
    monkeypatch.setattr(_fresh_log, "log_dir", lambda: "/dev/null/nope")
    monkeypatch.setattr(_fresh_log, "log_path", lambda: "/dev/null/nope/x.log")
    assert _fresh_log.setup(to_stderr=False) is None
    _fresh_log.get_logger("probe").info("这条会被静默丢弃")   # 不抛
    assert "未启用" in _fresh_log.describe_target()


def test_logging_does_not_propagate_to_root(_fresh_log):
    """不往 root 冒泡，免得宿主程序被灌进本不属于它的记录。"""
    _fresh_log.setup(to_stderr=False)
    assert _fresh_log.get_logger().propagate is False


def test_log_dir_follows_the_frozen_convention(monkeypatch):
    """与结果池 / 逐 bar 缓存同一套：冻结写用户目录，开发写仓库内。"""
    import deltalab_log

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert deltalab_log.log_dir().startswith(os.path.expanduser("~"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert "data" in deltalab_log.log_dir()


def test_no_silent_stderr_prints_remain():
    """诊断不能再用 ``print(file=sys.stderr)``——冻结包里那是静默 no-op。

    用 AST 判断**真实的调用**，不做文本匹配：说明这件事的注释与 docstring
    里必然会出现这个写法，按文本扫会把它们一起报进来。
    ``tools/`` 下的命令行脚本不在范围内，它们本来就跑在有终端的地方。
    """
    import ast

    root = pathlib.Path(_REPO_ROOT)
    offenders = []
    files = list(root.glob("*.py")) + list((root / "deltalab_ui").glob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                continue
            for kw in node.keywords:
                if kw.arg != "file":
                    continue
                target = ast.unparse(kw.value)
                if "stderr" in target or "stdout" in target:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"这些地方仍在往终端静默打印（冻结包里没有终端）: {offenders}")


# --------------------------------------------------------------------------
#  19. 左侧表单的等宽对齐
# --------------------------------------------------------------------------

@pytest.mark.gui
def test_input_column_fits_the_longest_structure_name():
    """输入列必须装得下最长的结构名，否则大类/子类型只能单独放宽。

    打 ``gui`` 标记：要量字体宽度就得有真实 Tk 根窗口，而无窗口服务器时
    建根会让**整个进程 abort**——``try/except TclError`` 挡不住，必须靠
    分批（CI 的 GUI 批跑在 xvfb 下）。

    此前输入列 168px 装不下「敲出计零·区间固赔·到期杠杆累计」，那两行被
    做成跨两列——同一张表里出现两种宽度，看着就是没对齐。列宽是按字体实测
    定的，换字体后这条会先红。
    """
    tk = pytest.importorskip("tkinter")
    from tkinter import font as tkfont

    from deltalab_ui.constants import OPTION_CLASSES, SUBTYPE_DISPLAY
    from deltalab_ui.theme import FORM_INPUT_W

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示")
    root.withdraw()
    try:
        metric = tkfont.nametofont("TkDefaultFont")
        widest = max(
            list(SUBTYPE_DISPLAY.values()) + list(OPTION_CLASSES),
            key=metric.measure)
        # Combobox 的箭头与内边距实测 26px
        needed = metric.measure(widest) + 26
    finally:
        root.destroy()

    assert FORM_INPUT_W >= needed, (
        f"输入列 {FORM_INPUT_W}px 装不下 {widest!r}（需要 {needed}px）")


def test_class_and_subtype_share_the_common_input_column():
    """大类 / 子类型不能再跨列——它们要和下方各行等宽。"""
    import inspect

    import gui_app

    source = inspect.getsource(gui_app.BacktestApp._build_ui)
    for name in ("class_cb", "self._subtype_cb"):
        placed = [line for line in source.splitlines()
                  if f"_form_input({name}" in line]
        assert placed, f"没找到 {name} 的布局调用"
        assert not any("columnspan" in line for line in placed), (
            f"{name} 又被放宽成跨列了——同一张表会出现两种输入宽度")


@pytest.mark.parametrize("subtype", _DE_SUBTYPES)
@pytest.mark.parametrize("cp", [1, -1])
def test_zero_payouts_are_a_valid_contract(subtype, cp):
    """区间赔付 / 熔断赔付填 0 必须能正常定价。

    0 是真实条款：区间赔付 0 = 标的落在 K~H 区间那几天不结算，熔断赔付
    0 = 熔断之后不再有现金流。建构闭包一度写成 ``p["fix"] if p["fix"]
    else None``——0.0 是 falsy，被悄悄转成"未填写"，于是填 0 直接报错。
    """
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    params = {name: default
              for name, _label, _dtype, default, *_ in cfg["params"]}
    params.update(s0=100.0, cp=cp, fix=0.0, P=0.0, amount=0.0)
    option = cfg["build"](subtype, params)
    option.nPath = 400
    assert np.isfinite(option.get_price())


def test_default_params_price_every_decumulator_subtype():
    """默认参数（区间赔付与熔断赔付都是 0）下 13 个子类型全部可定价。"""
    from deltalab_ui.constants import OPTION_CLASSES

    cfg = OPTION_CLASSES["累计期权 (Decumulator)"]
    params = {name: default
              for name, _label, _dtype, default, *_ in cfg["params"]}
    params["s0"] = 100.0
    assert params["fix"] == 0.0 and params["amount"] == 0.0
    for subtype in cfg["subtypes"]:
        option = cfg["build"](subtype, params)
        option.nPath = 400
        assert np.isfinite(option.get_price()), f"{subtype} 在默认参数下不能定价"


def test_build_closure_does_not_coerce_falsy_payouts():
    """守卫：建构闭包不能再把 0 当成"未填写"。"""
    import inspect

    from deltalab_ui import constants

    source = inspect.getsource(constants)
    for field in ("fix", "P", "amount"):
        bad = f'{field}=p["{field}"] if p["{field}"] else None'
        assert bad not in source, (
            f"{field} 又被做了 falsy 转换——0 会被当成未填写")
