# _*_ coding: utf-8 _*_
"""结果池：快照的构造、持久化与增删。

一条"保留的回测结果"从生成到落盘再到重放，整条链路都在这里——把一次运行冻结
成 ``SavedBacktestResult``（含四组可比签名与一份重放配方）、写进
``backtest_pool_store``、载入时反序列化回来、以及按需拿重放配方重跑出 bar 级
明细。结果池**长什么样**（表格、勾选、右键菜单）不在这里，在
``deltalab_ui.view_compare``。

这一组里十八个方法原本就是 ``@staticmethod``，而且测试大量以
``BacktestApp._x(SimpleNamespace(...), ...)`` 的形式传假 self 调用它们。整组放
进同一个 Mixin 而不是拆成模块函数，正是为了保住这个调法：Mixin 在 MRO 上，
``BacktestApp._x(...)`` 照常解析得到，组内静态方法之间也能互相直呼。

``ComparisonMixin`` 的两个方法（``_persist_pool_view`` /
``_saved_pool_actions_allowed``）在这里以显式类级调用访问。写成 ``self._x()``
会在假 self 上炸——那些 SimpleNamespace 只提供数据属性，不提供方法。
"""

import copy
import datetime
import hashlib
import os
from dataclasses import dataclass, field
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np

import backtest_pool_store
import history_bar_cache
import history_selection
from history_selection import DEFAULT_FIXED_TIMES, HISTORY_PERIOD_DEFS
from pricing import (
    CloseToCloseStrategy,
    FixedTimeStrategy,
    HedgeBacktest,
    HedgeBandStrategy,
    result_daily_frame,
    summarize_strategy_result,
)
from pricing.hedge_backtest import (
    _infer_intraday_steps,
    _rescale_option_to_real_s0,
    _rescale_strategy_to_real_s0,
    format_band_value,
)

from deltalab_ui import formatting, snapshot_detail, wind_resolve
from deltalab_ui.constants import (
    BASELINE_STRATEGY_STYLE_KEY,
    OPTION_CLASSES,
    SNAPSHOT_ORIGIN_DISPLAY,
    SNAPSHOT_ORIGIN_HISTORY_REPLAY,
    SNAPSHOT_ORIGIN_MANUAL,
    STRATEGY_DISPLAY,
    SUBTYPE_DISPLAY,
)
from deltalab_ui.view_compare import ComparisonMixin


@dataclass
class SavedBacktestResult:
    """回测结果池里的轻量快照：只保存对比所需字段，外加一份重放配方。

    展示层（``summary_row`` / ``daily_frame`` / 四个签名 key / ``form_state``）
    恒可用，重开程序后画曲线、算差异、应用策略都只读它。``replay`` 是可选
    的第二层，存的是**输入**（行情切片 + 冻结的运行参数），供「加载明细」
    重跑出 bar 级结果——bar 级**输出**平均 971 KB，那个不存。
    """

    result_id: str
    name: str
    saved_at: datetime.datetime
    summary_row: dict
    daily_frame: object
    strategy_label: str
    parameter_summary: str
    source_label: str
    option_label: str
    path_key: tuple
    # 四组可比属性的签名，都保留键名，因此能逐字段 diff 出"差在哪一项"。
    # path_key 是价格数组的哈希，只能判"数据变没变"，定位不了原因。
    market_key: tuple
    contract_key: tuple
    economics_key: tuple
    position: int
    # 跨页配色键：曲线名是用户可改的结果名，不能用来对齐策略优选页的颜色。
    style_key: str = BASELINE_STRATEGY_STYLE_KEY
    # 结构化来源，不随重命名丢失；origin_meta 记录优选周期、批次与当时排名。
    origin: str = SNAPSHOT_ORIGIN_MANUAL
    origin_meta: dict = field(default_factory=dict)
    # 本次运行的对冲策略输入，按原始单位保存，供回填左侧表单继续调参。
    # parameter_summary 是给人看的一句话，回不了表单。
    form_state: dict = field(default_factory=dict)
    # 保存顺序的权威来源。对比表原先按 result_id 字典序排，等于把「顺序」
    # 焊死在 ID 格式上：改 ID 方案会静默乱序，第 10000 条也会排到 9999 前。
    sequence: int = 0
    # 重放配方；缺失（旧包、或行情拿不到）时「加载明细」置灰而不是假装成功。
    replay: dict = field(default_factory=dict)
    # 下拉里选的那个 bar 粒度原值（可能是「自动（推荐）」）。纯展示字段，
    # **不进任何签名**：market_key 里的 wind_bar_size 已经是解析后的实际粒度，
    # 签名要的是"实际跑的是什么"。把它加进 market_key 会改变签名的键集合，
    # 于是新旧快照混选时 _differing_field_names 两侧键集合不同、整体退回属性
    # 级——差异比对会被这个纯展示需求悄悄削弱一档。
    bar_size_requested: str = ""
    # 引擎实际用的每交易日 bar 数。同样是纯展示字段、同样不进签名：
    # economics_key 里的 steps_per_day 记的是 GUI **传进去**的值，而真实行情
    # 下那是占位 1（见 _gui_steps_per_day），引擎自己再从时间索引推断。0 表示
    # 本字段上线前保留的快照没有记。
    intraday_steps: int = 0
    # 该快照在磁盘上的位置，载入时回填；重命名原地覆盖、删除据此删文件。
    store_path: str = ""

class SnapshotStoreMixin:
    """结果池的构造与持久化；混入 ``BacktestApp``，不单独实例化。"""

    def _sync_retain_button_state(self):
        button = getattr(self, "_retain_btn", None)
        if button is None:
            return
        can_retain = (
            getattr(self, "_active_job", None) is None
            and getattr(self, "_latest_backtest", None) is not None
            and getattr(self, "_latest_retained_result_id", None) is None
        )
        button.configure(state="normal" if can_retain else "disabled")

    @staticmethod
    def _backtest_path_key(bt, gui_state):
        """对实际价格、时间索引和交易日分组生成会话内可比签名。"""
        result = getattr(bt, "_results", {}) or {}
        prices = np.ascontiguousarray(
            np.asarray(result.get("prices", getattr(bt, "prices", [])),
                       dtype="<f8")
        )
        digest = hashlib.sha256()
        source = str(gui_state.get("source", "unknown"))
        digest.update(source.encode("utf-8"))
        digest.update(prices.tobytes())

        groups = result.get("trading_day_groups")
        if groups is not None:
            group_array = np.ascontiguousarray(np.asarray(groups, dtype="<i8"))
            digest.update(b"groups")
            digest.update(group_array.tobytes())

        timestamps = getattr(bt, "timestamps", None)
        if timestamps is not None:
            try:
                import pandas as pd
                # run() 可能因敲出提前终止；签名必须和结果价格使用同一前缀，
                # 不能把未参与本次结果的尾部时间戳混入比较上下文。
                timestamp_index = pd.DatetimeIndex(timestamps)[:len(prices)]
                timestamp_values = np.ascontiguousarray(
                    np.asarray(timestamp_index.asi8, dtype="<i8"))
                digest.update(b"timestamps")
                digest.update(timestamp_values.tobytes())
            except Exception:
                digest.update(
                    repr(tuple(timestamps)[:len(prices)]).encode("utf-8"))
        return source, int(len(prices)), digest.hexdigest()

    @staticmethod
    def _snapshot_origin_from_state(gui_state):
        """从单次回测状态判断快照来源。

        历史分段重放会在状态里留下 history_replay_* 字段；手工点『保留当前
        结果』时据此把它标成分段重放，而不是笼统的手工回测。
        """
        gui_state = dict(gui_state or {})
        strategy = str(
            gui_state.get("history_replay_strategy", "") or "").strip()
        if not strategy:
            return SNAPSHOT_ORIGIN_MANUAL, {}
        lookback = str(gui_state.get("history_replay_lookback", "") or "")
        return SNAPSHOT_ORIGIN_HISTORY_REPLAY, {
            "lookback": lookback,
            "period_label": dict(HISTORY_PERIOD_DEFS).get(lookback, lookback),
            "history_strategy": strategy,
            "window_id": str(
                gui_state.get("history_replay_window_id", "") or ""),
        }

    @staticmethod
    def _snapshot_style_key(gui_state):
        """从单次回测状态取配色键；σ 一律换算成日波动倍数。"""
        strategy = str(gui_state.get("strategy_name", "")).strip()
        if strategy != "hedge_band":
            return formatting.strategy_style_key(
                strategy, fixed_times=gui_state.get("fixed_times"))
        band_type = gui_state.get("interval_type", "absolute")
        threshold = history_selection.finite_value(
            gui_state.get("price_interval"))
        sigma = None
        if threshold is not None:
            try:
                params = gui_state.get("params", {})
                sigma = HedgeBandStrategy.convert_threshold(
                    threshold, band_type, float(params["s0"]),
                    float(params["sigma"]),
                )["sigma"]
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                # 换算不出 σ 时退回按输入口径区分，绝不让不同带宽共享颜色。
                return formatting.strategy_style_key(
                    f"hedge_band_{band_type}_{threshold:.6g}")
        return formatting.strategy_style_key("hedge_band", sigma=sigma)

    @staticmethod
    def _strategy_snapshot_labels(gui_state):
        strategy = gui_state.get("strategy_name", "unknown")
        fallback_label = (
            "收盘兜底：开启"
            if bool(gui_state.get("force_day_close_hedge", False))
            else "收盘兜底：关闭"
        )

        def _with_fallback(parameters):
            return f"{parameters}；{fallback_label}"

        if strategy == "close_to_close":
            return "每日收盘", _with_fallback("每交易日最后一个采样点")
        if strategy == "fixed_times":
            times = str(gui_state.get("fixed_times", "")).strip() or "—"
            return "固定时刻", _with_fallback(times)
        if strategy == "hedge_band":
            band_type = gui_state.get("interval_type", "absolute")
            threshold = float(gui_state.get("price_interval", 0.0))
            if band_type == "relative":
                primary = f"相对 {threshold:.4%}"
            elif band_type == "sigma":
                primary = f"{threshold:g} 倍波动率（日波动）"
            else:
                primary = f"绝对 {threshold:g}"
            try:
                params = gui_state.get("params", {})
                converted = HedgeBandStrategy.convert_threshold(
                    threshold, band_type, float(params["s0"]),
                    float(params["sigma"]),
                )
                equivalents = (
                    f"绝对 {format_band_value(converted['absolute'])} / "
                    f"相对 {converted['relative']:.4%} / "
                    f"{format_band_value(converted['sigma'])}σ"
                )
                return "固定间隔", _with_fallback(
                    f"{primary}；等价 {equivalents}")
            except (KeyError, TypeError, ValueError):
                return "固定间隔", _with_fallback(primary)
        return (
            STRATEGY_DISPLAY.get(strategy, str(strategy)),
            _with_fallback("—"),
        )

    @staticmethod
    def _snapshot_comparison_data(result):
        """在入池时缓存指标与日级曲线，避免保留和重复聚合 bar 级数组。"""
        summary_row = summarize_strategy_result(result, "snapshot")
        daily_frame = result_daily_frame(result).copy(deep=True)
        return copy.deepcopy(summary_row), daily_frame

    @staticmethod
    def _snapshot_replay_recipe(bt, gui_state):
        """把重跑这次回测所需的**输入**摘成配方。

        存价格序列本身，而不是「来源标识 + 重新取数」：
        - 模拟的路径由 ``real_vol`` 参与生成，而 ``real_vol`` 从来没进过快照；
        - CSV 文件可能被移动或删除；
        - Wind 非期货走前复权，复权因子随时间变，同一区间换个日期取回来的
          **不是同一串数**。
        存下来就都不依赖了，离线也能精确重现。

        固定时刻存的是 ``effective_times``（已按交易时段剔掉休市目标）而不是
        用户填的原值：重放时拿不到品种 session 元数据，用原值会触发逐日严格
        校验而失败——原始运行明明是成功的。
        """
        import pandas as pd
        results = getattr(bt, "_results", None) or {}
        prices = np.asarray(results.get("prices", ()), dtype=float)
        if prices.size < 2:
            return {}
        timestamps = results.get("timestamps")
        index = None
        if timestamps is not None:
            try:
                index = [pd.Timestamp(ts).isoformat()
                         for ts in pd.DatetimeIndex(timestamps)]
            except (TypeError, ValueError):
                index = None
        if index is not None and len(index) != prices.size:
            index = None
        recipe = {
            "prices": prices.tolist(),
            "index": index,
            "steps_per_day": int(results.get("steps_per_day", 1) or 1),
            "source": gui_state.get("source"),
            "cls_name": gui_state.get("cls_name"),
            "subtype": gui_state.get("subtype"),
            "params": copy.deepcopy(dict(gui_state.get("params", {}))),
            "position": wind_resolve.normalize_position(
                gui_state.get("position")),
            "quantity": gui_state.get("quantity"),
            "multiplier": gui_state.get("multiplier"),
            "tc_rate": gui_state.get("tc_rate"),
            "slippage_bps": gui_state.get("slippage_bps", 0.0),
            "force_day_close_hedge": bool(
                gui_state.get("force_day_close_hedge", False)),
            "evaluation_days": results.get("evaluation_days"),
            "form_state": SnapshotStoreMixin._snapshot_form_state(gui_state),
            "fixed_times_effective": list(
                results.get("fixed_time_effective_times", ()) or ()),
            # 摘要页直接读 bt._gui_meta，重放出来的对象也得有一份。
            "gui_meta": copy.deepcopy(dict(getattr(bt, "_gui_meta", {}) or {})),
        }
        return recipe

    @staticmethod
    def _strategy_from_form_state(form_state, *, fixed_times_effective=None):
        """按冻结的表单输入重建策略对象。

        固定时刻优先用 ``effective_times``：重放时没有品种交易时段元数据，
        用用户填的原值会让休市目标重新参与逐日严格校验而失败——而原始运行
        是成功的，那一步已经把它们剔掉了。
        """
        name = str(form_state.get("strategy_name", "close_to_close"))
        if name == "close_to_close":
            return CloseToCloseStrategy()
        if name == "fixed_times":
            times = fixed_times_effective or form_state.get(
                "fixed_times", DEFAULT_FIXED_TIMES)
            return FixedTimeStrategy(times)
        if name == "hedge_band":
            return HedgeBandStrategy(
                band_type=form_state.get("interval_type", "absolute"),
                threshold=form_state.get("price_interval", 1.0),
                sigma_source=form_state.get("sigma_source", "implied"),
                window_days=form_state.get("sigma_window", 20),
            )
        raise ValueError(f"未知对冲策略: {name}")

    def _replay_saved_snapshot(self, snapshot):
        """取这条快照的逐 bar 明细，返回 ``(回测对象, 是否重算过)``。

        **正常路径是读已保存的数据，不是重跑。** 保留结果的那一刻 bar 级
        结果就已落盘（见 ``_make_saved_backtest_result``），这里按同一份配
        方算出的 key 读回来，实测 3 ms；真跑一次要几百毫秒。

        只有缓存确实不在时才回退重算——缓存被清过、换了机器、或本功能上线
        前保留的旧快照。回退与「保留结果」时那次运行同源：价格序列是存下来
        的原始切片，期权与策略按同一个 ``_rescale_option_to_real_s0`` 重定
        基，因此逐 bar 明细与当时一致，不依赖 CSV 文件是否还在、也不依赖
        Wind 能否联网。调用方要把「重算过」说出来：用户按的是「加载」。
        """
        import pandas as pd
        recipe = dict(getattr(snapshot, "replay", None) or {})
        if not recipe:
            raise ValueError(
                "这条结果没有重放配方（可能保存于本功能上线之前），"
                "无法加载明细；重新跑一次并保留即可。")
        prices = np.asarray(recipe.get("prices", ()), dtype=float)
        if prices.size < 2:
            raise ValueError("重放配方里的行情序列不完整。")
        index = recipe.get("index")
        series = (pd.Series(prices, index=pd.to_datetime(index))
                  if index else pd.Series(prices))

        cls_name = recipe.get("cls_name")
        cfg = OPTION_CLASSES.get(cls_name)
        if cfg is None:
            raise ValueError(f"未知期权大类：{cls_name}")
        option = cfg["build"](recipe.get("subtype"), dict(recipe.get("params", {})))
        strategy = SnapshotStoreMixin._strategy_from_form_state(
            dict(recipe.get("form_state", {})),
            fixed_times_effective=recipe.get("fixed_times_effective") or None)

        # 真实行情当时按首价整体缩放过期权与绝对带宽，重放必须走同一步，
        # 否则行权价与带宽会停在参考价口径上，明细和当时对不上。
        if str(recipe.get("source")) in ("csv", "wind"):
            option, info = _rescale_option_to_real_s0(
                option, float(series.iloc[0]))
            strategy = _rescale_strategy_to_real_s0(strategy, info["ratio"])

        bt = HedgeBacktest(
            option, series,
            tc_rate=recipe.get("tc_rate", 0.0),
            position=recipe.get("position", 1),
            quantity=recipe.get("quantity", 1.0),
            multiplier=recipe.get("multiplier", 5),
            strategy=strategy,
            steps_per_day=int(recipe.get("steps_per_day", 1) or 1),
            slippage_bps=recipe.get("slippage_bps", 0.0),
            force_day_close_hedge=bool(
                recipe.get("force_day_close_hedge", False)),
            evaluation_days=recipe.get("evaluation_days"),
        )
        bt._gui_meta = dict(recipe.get("gui_meta", {})) or {
            "cls_name": cls_name,
            "subtype": recipe.get("subtype"),
            "source": recipe.get("source"),
        }
        cached = history_bar_cache.load_recipe(recipe)
        if cached is not None:
            # run() 只设置 _results（hedge_backtest 里唯一一处赋值），所以
            # 把结果塞进未运行的对象与真跑出来的等价。
            bt._results = cached
            return bt, False
        bt.run()
        history_bar_cache.store_recipe(recipe, dict(bt._results or {}))
        return bt, True

    def _load_saved_snapshot_detail(self):
        """右键菜单「加载明细」：重跑选中快照并渲染到结果页。"""
        if not ComparisonMixin._saved_pool_actions_allowed(self):
            return
        result_id = self._focused_saved_backtest_id()
        if result_id not in self._saved_backtests:
            messagebox.showinfo("请选择结果", "请先在结果池中点击一条结果。")
            return
        snapshot = self._saved_backtests[result_id]
        try:
            bt, recomputed = self._replay_saved_snapshot(snapshot)
        except Exception as exc:                              # noqa: BLE001
            messagebox.showerror("加载明细失败", str(exc))
            return
        self._latest_backtest = bt
        self._show_results(bt)
        if recomputed:
            # 菜单项叫「加载明细」，正常就该是秒开的读盘。走到重算说明保存
            # 的明细不在了，得说一声，否则用户只感觉「怎么卡了一下」。
            self._set_status(
                f"已保存的明细不存在，已按配方重新计算并补存：{snapshot.name}")
        else:
            self._set_status(
                f"已加载『{snapshot.name}』的明细  |  "
                "回测摘要 / 对冲图表 / 每日明细 / 波动率分析均为该次结果")

    @staticmethod
    def _saved_snapshot_strategy_type(snapshot):
        """读取不可被用户重命名影响的快照策略类型。"""
        summary_row = getattr(snapshot, "summary_row", {}) or {}
        return str(summary_row.get("strategy_type", "")).strip()

    @staticmethod
    def _saved_snapshot_origin_label(snapshot):
        """产生方式列文案。

        来源读结构化的 origin 字段，用户重命名结果不会让它丢失。
        """
        origin = str(getattr(snapshot, "origin", "") or
                     SNAPSHOT_ORIGIN_MANUAL)
        return SNAPSHOT_ORIGIN_DISPLAY.get(origin, origin)

    @staticmethod
    def _snapshot_origin_detail(snapshot):
        """把快照的结构化来源摊成一句可读的溯源说明。

        分段重放会记下它回放的是哪个周期、哪个候选、哪一段。溯源不参与任何
        排名，它只回答"这条结果是怎么来的"。
        """
        meta = getattr(snapshot, "origin_meta", None) or {}
        parts = []
        period = str(meta.get("period_label", "") or "").strip()
        if period:
            parts.append(f"周期 {period}")
        strategy = str(meta.get("history_strategy", "") or "").strip()
        if strategy:
            parts.append(f"候选『{strategy}』")
        window_id = str(meta.get("window_id", "") or "").strip()
        if window_id:
            parts.append(f"分段 {window_id}")
        return "  ·  ".join(parts)

    @staticmethod
    def _make_saved_backtest_result(
            bt, gui_state, result_id, name, saved_at=None, *,
            origin=SNAPSHOT_ORIGIN_MANUAL, origin_meta=None, sequence=0):
        snapshot_position = wind_resolve.normalize_position(
            gui_state.get("position"))
        result_position = (getattr(bt, "_results", {}) or {}).get("position")
        if result_position is not None:
            result_position = wind_resolve.normalize_position(result_position)
            if result_position != snapshot_position:
                raise ValueError(
                    "回测结果的头寸方向与本次左侧参数不一致，已拒绝保存。")
        summary_row, daily_frame = SnapshotStoreMixin._snapshot_comparison_data(
            bt._results)
        strategy_label, parameter_summary = (
            SnapshotStoreMixin._strategy_snapshot_labels(gui_state))
        path_key = SnapshotStoreMixin._backtest_path_key(bt, gui_state)
        market_key, contract_key, economics_key = (
            snapshot_detail.signature_keys_from_state(
                gui_state,
                data_digest=path_key[2],
                steps_per_day=bt._results.get("steps_per_day"),
            ))
        subtype = gui_state.get("subtype", "—")
        replay_recipe = SnapshotStoreMixin._snapshot_replay_recipe(bt, gui_state)
        # 保留的这一刻 bar 级结果还在手里，顺手落盘：之后点「加载明细」就
        # 不必按配方重跑（实测 3 ms vs 620 ms）。与策略优选共用同一套内容
        # 寻址缓存，写失败只是白跑一次，不影响保留本身。
        history_bar_cache.store_recipe(
            replay_recipe, dict(getattr(bt, "_results", None) or {}))
        return SavedBacktestResult(
            result_id=result_id,
            name=name,
            saved_at=saved_at or datetime.datetime.now(),
            summary_row=summary_row,
            daily_frame=daily_frame,
            strategy_label=strategy_label,
            parameter_summary=parameter_summary,
            source_label=formatting.snapshot_source_label(gui_state),
            option_label=SUBTYPE_DISPLAY.get(subtype, str(subtype)),
            path_key=path_key,
            market_key=market_key,
            contract_key=contract_key,
            economics_key=economics_key,
            position=snapshot_position,
            style_key=SnapshotStoreMixin._snapshot_style_key(gui_state),
            origin=str(origin or SNAPSHOT_ORIGIN_MANUAL),
            origin_meta=copy.deepcopy(dict(origin_meta or {})),
            form_state=SnapshotStoreMixin._snapshot_form_state(gui_state),
            sequence=int(sequence or 0),
            replay=replay_recipe,
            bar_size_requested=str(
                gui_state.get("wind_bar_size_requested", "") or ""),
            intraday_steps=SnapshotStoreMixin._effective_intraday_steps(bt),
        )

    @staticmethod
    def _effective_intraday_steps(bt):
        """本次回测**实际**的每交易日 bar 数。

        不能只读 ``_results["steps_per_day"]``：那是 GUI 传给引擎的值，而
        ``_gui_steps_per_day`` 对真实行情一律返回占位 1（CSV 分支干脆传
        ``None``），引擎再自己从时间索引推断并取两者较大值。照那个值展示，
        一条 15 分钟 Wind 回测会写着「日内采样 1」，而它每天实际十几根 bar。

        推不出来时退回传入值——模拟路径的索引常常不是 DatetimeIndex，而它那
        个值本来就是用户直接填的真实采样密度。
        """
        results = getattr(bt, "_results", None) or {}
        declared = max(1, int(results.get("steps_per_day", 1) or 1))
        timestamps = getattr(bt, "timestamps", None)
        if timestamps is None or len(timestamps) < 2:
            return declared
        try:
            inferred = int(_infer_intraday_steps(timestamps))
        except (AttributeError, TypeError, ValueError):
            return declared
        return max(declared, inferred)

    @staticmethod
    def _snapshot_form_state(gui_state):
        """摘出可回填左侧表单的对冲策略输入，保持运行时的原始单位。

        只取对冲策略这一组：期权结构、方向、成本和行情来源改动面太大，回填
        它们等于把整张表单换掉，而用户点这个按钮想做的是接着这条结果继续调
        对冲参数。
        """
        gui_state = dict(gui_state or {})
        return {
            "strategy_name": str(gui_state.get("strategy_name", "") or ""),
            "fixed_times": str(gui_state.get("fixed_times", "") or ""),
            "interval_type": str(
                gui_state.get("interval_type", "") or "absolute"),
            "price_interval": history_selection.finite_value(
                gui_state.get("price_interval")),
            "force_day_close_hedge": bool(
                gui_state.get("force_day_close_hedge", False)),
            # 波动率口径决定带宽怎么换算，实际改变回测结果（它直接传给
            # HedgeBandStrategy）。此前它不在快照的任何一处：两条只差这个
            # 口径的结果，逐字段比下来会是"完全一致"，而数字明明不同。
            "sigma_source": str(
                gui_state.get("sigma_source", "") or "implied"),
            "sigma_window": history_selection.safe_int(
                gui_state.get("sigma_window"), 20),
        }

    @staticmethod
    def _validate_saved_result_name(saved_results, name, exclude_id=None):
        cleaned = str(name or "").strip()
        if not cleaned:
            raise ValueError("结果名称不能为空")
        if any(
                snapshot.result_id != exclude_id and snapshot.name == cleaned
                for snapshot in saved_results.values()):
            raise ValueError(f"结果名称已存在：{cleaned}")
        return cleaned

    _COMPARE_TAB_TITLE = " 🆚 结果对比 "

    def _update_saved_result_count(self):
        """把结果池条数写进标签页标题。

        空池不写 `(0)`：那时标签页里的占位符已经在说"先保留结果"，标题再挂
        一个 0 只是噪声。非空时数字要一直可见——它是"有没有东西可对比"的
        唯一常驻提示，此前挂在左侧按钮上。
        """
        notebook = getattr(self, "_nb", None)
        tab = getattr(self, "_compare_tab", None)
        if notebook is None or tab is None:
            return
        count = len(getattr(self, "_saved_backtests", None) or {})
        title = SnapshotStoreMixin._COMPARE_TAB_TITLE
        if count:
            title = f"{title.rstrip()} ({count}) "
        try:
            notebook.tab(tab, text=title)
        except tk.TclError:
            # 标签页已随窗口销毁：计数只是展示，不该反过来打断清理流程。
            pass

    # ---- 结果池落盘 ----
    _POOL_FIELDS = (
        "result_id", "name", "saved_at", "summary_row", "daily_frame",
        "strategy_label", "parameter_summary", "source_label", "option_label",
        "path_key", "market_key", "contract_key", "economics_key", "position",
        "style_key", "origin", "origin_meta", "form_state", "sequence",
        "replay", "bar_size_requested", "intraday_steps",
    )

    @staticmethod
    def _snapshot_to_payload(snapshot):
        """快照 → 可写盘的 payload。``store_path`` 不入包（它就是包的位置）。"""
        body = {
            key: backtest_pool_store.encode(getattr(snapshot, key))
            for key in SnapshotStoreMixin._POOL_FIELDS
        }
        return {
            "schema_version": backtest_pool_store.POOL_SCHEMA_VERSION,
            "sequence": int(getattr(snapshot, "sequence", 0) or 0),
            "result_id": str(snapshot.result_id),
            "saved_at": snapshot.saved_at.isoformat(),
            "snapshot": body,
        }

    @staticmethod
    def _payload_to_snapshot(payload):
        """payload → 快照。缺字段用 dataclass 默认值补，不让旧包整条报废。"""
        body = dict(payload.get("snapshot") or {})
        values = {
            key: backtest_pool_store.decode(body[key])
            for key in SnapshotStoreMixin._POOL_FIELDS if key in body
        }
        # 结构中文名跟着 SUBTYPE_DISPLAY 重算，不用包里那份。``option_label``
        # 是**显示名**却落了盘，于是结构一改名，老包会一直显示旧名，同一张表
        # 里新旧两套名字并存——而重放配方里正好存着内部键，据此现算即可。
        # 配方为空的包（``prices.size < 2``，见 _snapshot_replay_recipe）没有
        # 内部键可依，那时才退回落盘的那份。
        replay = values.get("replay")
        subtype = str((replay or {}).get("subtype") or "") if isinstance(
            replay, dict) else ""
        if subtype in SUBTYPE_DISPLAY:
            values["option_label"] = SUBTYPE_DISPLAY[subtype]

        saved_at = values.get("saved_at")
        if not isinstance(saved_at, datetime.datetime):
            values["saved_at"] = datetime.datetime.fromisoformat(
                str(payload.get("saved_at")))
        else:
            values["saved_at"] = saved_at.to_pydatetime() if hasattr(
                saved_at, "to_pydatetime") else saved_at
        snapshot = SavedBacktestResult(**values)
        snapshot.store_path = str(payload.get("_path", "") or "")
        return snapshot

    def _persist_saved_snapshot(self, snapshot):
        """写盘并返回一句状态栏后缀。

        失败不回滚内存池——这一条结果是真跑出来的，扔掉它比留着更糟。但
        「已保留」四个字必须诚实：写不进去就当场说清楚，否则用户下次开机
        才发现，而那时已经无从追查。
        """
        try:
            path = backtest_pool_store.write_snapshot(
                SnapshotStoreMixin._snapshot_to_payload(snapshot), enforce=False)
        except Exception as exc:                              # noqa: BLE001
            snapshot.store_path = ""
            return f"（未能写入本机：{exc}）"
        snapshot.store_path = path
        # 先写入再淘汰，顺序不能反（与 _save_history_result 同一条约定）：反
        # 过来会在写失败时白删一份。自己调 enforce_limit 而不是让
        # write_snapshot 代劳，是因为要拿到被淘汰的清单——store 那边的约定
        # 就是「调用方应当把返回值报给用户」。
        try:
            evicted = backtest_pool_store.enforce_limit()
        except Exception:                                     # noqa: BLE001
            evicted = []
        return "并已保存到本机" + SnapshotStoreMixin._forget_evicted_snapshots(
            self, evicted)

    def _forget_evicted_snapshots(self, evicted_paths):
        """把已淘汰出磁盘的快照同步移出内存池，返回一句状态栏后缀。

        内存池此前不受 ``MAX_RESULTS`` 约束，只有磁盘受。存到第 21 条时盘上
        第 1 条已经被删，标签页却仍写着 (21)、对比页也仍列着它——用户重开程
        序才发现少了一条，而那时已无从追查。留在内存里还有第二个后果：重命
        名它会走 ``write_snapshot(path=...)`` 把文件原样写回来，目录随即又超
        出上限。

        删除不可逆，即使是按规则删的也必须说一声。
        """
        paths = {str(path) for path in (evicted_paths or ()) if path}
        if not paths:
            return ""
        dropped = [
            snapshot for snapshot in self._saved_backtests.values()
            if str(getattr(snapshot, "store_path", "") or "") in paths
        ]
        for snapshot in dropped:
            self._saved_backtests.pop(snapshot.result_id, None)
            self._saved_comparison_selection.discard(snapshot.result_id)
            if getattr(self, "_latest_retained_result_id", None) == (
                    snapshot.result_id):
                self._latest_retained_result_id = None
        if not dropped:
            return ""
        names = "、".join(str(snapshot.name) for snapshot in dropped[:3])
        more = f" 等 {len(dropped)} 条" if len(dropped) > 3 else ""
        return (f"；已达上限 {backtest_pool_store.MAX_RESULTS} 条，"
                f"淘汰最旧的「{names}」{more}")

    def _load_saved_pool(self):
        """启动时载入结果池，并把序号推到已有最大值之后。

        序号不推的话，载入 20 条旧结果后第一次「保留」仍会生成
        ``result-0001``，而入池是按 key 直接赋值——盘里那条会被静默顶掉。
        """
        try:
            payloads, skipped = backtest_pool_store.read_all()
        except Exception as exc:                              # noqa: BLE001
            self._saved_pool_load_error = f"载入已保存结果失败：{exc}"
            return
        self._saved_pool_load_error = ""
        max_sequence = 0
        for payload in payloads:
            try:
                snapshot = SnapshotStoreMixin._payload_to_snapshot(payload)
            except Exception as exc:                          # noqa: BLE001
                skipped.append(
                    (os.path.basename(str(payload.get("_path", ""))),
                     f"字段无法还原：{exc}"))
                continue
            self._saved_backtests[snapshot.result_id] = snapshot
            max_sequence = max(max_sequence, int(snapshot.sequence or 0))
        self._saved_backtest_sequence = max_sequence
        # 恢复上次显示的那几条。与结果本身分开存：它坏了最多回到「全部隐
        # 藏」，而空状态会写明 N 条都还在池里。默认不全选——重开就把二十条
        # 曲线一起铺上去，比让用户点一下更糟。
        restored = backtest_pool_store.read_view_state() & set(
            self._saved_backtests)
        self._saved_comparison_selection = restored
        if skipped:
            self._saved_pool_load_error = (
                f"有 {len(skipped)} 条已保存结果未能载入："
                + "；".join(f"{name}（{why}）" for name, why in skipped[:3])
                + ("…" if len(skipped) > 3 else ""))

    def _store_current_backtest(self, name, *,
                                origin=SNAPSHOT_ORIGIN_MANUAL,
                                origin_meta=None):
        """用已完成的普通回测结果创建快照，绝不重跑回测。"""
        bt = getattr(self, "_latest_backtest", None)
        gui_state = getattr(self, "_latest_backtest_state", None)
        if bt is None or gui_state is None:
            raise ValueError("没有可保留的已完成回测结果。")
        if getattr(self, "_latest_retained_result_id", None) is not None:
            raise ValueError("当前回测结果已经保留。")

        cleaned_name = self._validate_saved_result_name(
            self._saved_backtests, name)
        next_sequence = self._saved_backtest_sequence + 1
        result_id = f"result-{next_sequence:04d}"
        # 序号在载入结果池时已经推到已有最大值之后（见 _load_saved_pool），
        # 所以这里不会撞上盘里那些旧 ID。
        snapshot = self._make_saved_backtest_result(
            bt, gui_state, result_id, cleaned_name,
            origin=origin, origin_meta=origin_meta, sequence=next_sequence)
        self._saved_backtest_sequence = next_sequence
        self._saved_backtests[result_id] = snapshot
        self._saved_comparison_selection.add(result_id)
        self._latest_retained_result_id = result_id
        write_note = self._persist_saved_snapshot(snapshot)
        ComparisonMixin._persist_pool_view(self)
        self._update_saved_result_count()
        self._sync_retain_button_state()
        self._refresh_saved_comparison_if_visible()
        self._set_status(
            f"已保留『{cleaned_name}』{write_note}  |  可修改策略或参数继续回测"
            f"（共 {len(self._saved_backtests)} 条）")
        return snapshot

    @staticmethod
    def _next_default_name_number(saved_results, base):
        """同前缀名字接着往下编号：池里已有『X #05』就返回 6。

        默认编号不能用入池序号 ``_saved_backtest_sequence``：那是整个结果池
        的保存次序，中间存过别的策略就会跳号——存到「每日收盘 #05」后再存
        十条固定间隔，下一条每日收盘会默认成 #16。用户读这个数字时问的是
        「这是我第几次存这个策略」，所以只数同前缀的那几条。

        只认前缀完全一致、后缀是纯 ASCII 数字的名字：用户改过的名字（如
        「每日收盘 最终版」）不参与编号，也就不会把编号顶到莫名其妙的位置。
        """
        prefix = f"{base} #"
        highest = 0
        for snapshot in saved_results.values():
            name = str(getattr(snapshot, "name", "") or "")
            if not name.startswith(prefix):
                continue
            tail = name[len(prefix):]
            # 光 isdigit() 会放过上标之类的字符，而 int() 认不出它们。
            if tail.isascii() and tail.isdigit():
                highest = max(highest, int(tail))
        return highest + 1

    def _retain_current_backtest(self):
        bt = getattr(self, "_latest_backtest", None)
        gui_state = getattr(self, "_latest_backtest_state", None)
        if bt is None or gui_state is None:
            messagebox.showinfo("没有可保留结果", "请先成功运行一次回测。")
            return
        if getattr(self, "_latest_retained_result_id", None) is not None:
            messagebox.showinfo("结果已保留", "当前回测结果已经在对比结果池中。")
            return

        strategy_label, parameters = self._strategy_snapshot_labels(gui_state)
        short_parameters = parameters.split("；", 1)[0]
        if gui_state.get("strategy_name") == "close_to_close":
            base_name = strategy_label
        else:
            base_name = f"{strategy_label} · {short_parameters}"
        next_number = SnapshotStoreMixin._next_default_name_number(
            self._saved_backtests, base_name)
        default_name = f"{base_name} #{next_number:02d}"
        while True:
            name = simpledialog.askstring(
                "保留回测结果",
                "为本次结果命名（可稍后在对比页重命名）：",
                initialvalue=default_name, parent=self,
            )
            if name is None:
                return
            try:
                name = self._validate_saved_result_name(
                    self._saved_backtests, name)
                break
            except ValueError as exc:
                messagebox.showerror("名称无效", str(exc))

        origin, origin_meta = SnapshotStoreMixin._snapshot_origin_from_state(
            gui_state)
        try:
            self._store_current_backtest(
                name, origin=origin, origin_meta=origin_meta)
        except Exception as exc:
            messagebox.showerror("保留结果失败", str(exc))
            return

    @staticmethod
    def _saved_comparison_payload(snapshots):
        """只读取已完成快照生成对比数据，绝不重新运行回测。

        本页展示的是每条结果自己的绝对指标，不设固定基准、不算相对改善：
        谁跟谁比、比哪一项，交给排序和并列的曲线去回答。
        """
        import pandas as pd
        snapshots = list(snapshots)
        rows = []
        daily_curves = {}
        for snapshot in snapshots:
            if snapshot.name in daily_curves:
                raise ValueError(f"保存结果名称重复：{snapshot.name}")
            description = (
                f"{snapshot.strategy_label} · {snapshot.parameter_summary} · "
                f"{snapshot.option_label} · {snapshot.source_label}"
            )
            row = copy.deepcopy(snapshot.summary_row)
            row.update({
                "strategy": snapshot.name,
                "meta_description": description,
                "meta_result_id": snapshot.result_id,
                # 保存顺序的权威来源。别退回按 result_id 字典序排：那把顺序
                # 焊死在 ID 格式上，第 10000 条会排到 9999 前面，跨会话载入
                # 后若换了 ID 方案更会静默乱序——而乱序不会有任何断言挂掉。
                "meta_sequence": int(getattr(snapshot, "sequence", 0) or 0),
                # 曲线名可被用户改写，配色必须走稳定的策略身份键。
                "meta_style_key": getattr(
                    snapshot, "style_key", BASELINE_STRATEGY_STYLE_KEY),
                "meta_origin": getattr(
                    snapshot, "origin", SNAPSHOT_ORIGIN_MANUAL),
                "meta_origin_detail": (
                    SnapshotStoreMixin._snapshot_origin_detail(snapshot)),
            })
            rows.append(row)
            daily_curves[snapshot.name] = snapshot.daily_frame
        summary = pd.DataFrame(rows)
        if not summary.empty:
            # 默认按保存顺序，与结果池上下对照着看最省事；想看谁高谁低，
            # 点一下列头即可。按某个指标排序会隐含"这一列越高/越低越好"的
            # 暗示，而本页不替用户下这个判断。
            summary = summary.sort_values(
                ["meta_sequence", "meta_result_id"],
                kind="stable").reset_index(drop=True)
        return summary, daily_curves

    def _rename_saved_backtest(self, result_id, new_name):
        snapshot = self._saved_backtests[result_id]
        snapshot.name = SnapshotStoreMixin._validate_saved_result_name(
            self._saved_backtests, new_name, exclude_id=result_id)
        # 原地覆盖同一个文件：文件名带序号，是载入顺序的依据，不能因为改
        # 展示名就换名字。
        if snapshot.store_path:
            try:
                backtest_pool_store.write_snapshot(
                    SnapshotStoreMixin._snapshot_to_payload(snapshot),
                    path=snapshot.store_path, enforce=False)
            except Exception as exc:                          # noqa: BLE001
                # 改名失败确实不该打断本次会话——内存里的名字已经改好了，
                # 曲线、对比表、状态栏都会用新名字。但也不能像以前那样一声
                # 不吭：只读目录或文件被占用时，用户看到「已重命名」、重启
                # 后却发现名字变了回去，而中间没有任何一处提示过。抛出去交
                # 给调用方措辞，内存状态保持已改。
                raise OSError(
                    f"结果名已在本次会话生效，但没能写入结果文件"
                    f"（{exc}）；重启后会恢复为旧名称。") from exc
        return snapshot

    def _delete_saved_backtest(self, result_id):
        """删掉一条已保留结果；文件删不掉时原样保留并返回 None。

        顺序是**先删文件、再动内存**。反过来（旧写法先 pop）在只读目录或
        Windows 文件被占用时会走成最坏的一种：这条结果从界面上消失、状态栏
        写着「已删除」，文件却还躺在盘上，下次启动它自己又回来了。删不掉就
        当没删过，把这条留在池子里，由调用方说清楚。
        """
        snapshot = self._saved_backtests[result_id]
        # ``delete_snapshot`` 对「本来就不存在」和「删不掉」都返回 False，而
        # 这两件事对池子的意义正好相反：前者目标状态已经达成，该照常出池。
        # 所以真正的失败判据是「没删成 **且** 文件还在」。
        if (snapshot.store_path
                and not backtest_pool_store.delete_snapshot(snapshot.store_path)
                and os.path.exists(snapshot.store_path)):
            return None
        self._saved_backtests.pop(result_id)
        self._saved_comparison_selection.discard(result_id)
        if getattr(self, "_latest_retained_result_id", None) == result_id:
            self._latest_retained_result_id = None
        ComparisonMixin._persist_pool_view(self)
        SnapshotStoreMixin._update_saved_result_count(self)
        SnapshotStoreMixin._sync_retain_button_state(self)
        return snapshot
