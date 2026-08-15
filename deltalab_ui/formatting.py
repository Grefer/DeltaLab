# _*_ coding: utf-8 _*_
"""跨页共用的小格式化函数。

放这里的门槛是"三个以上互不相干的调用点"：结果页的明细表、结果池的来源列、
策略优选的重放窗口标签都要把同一个值渲染成同一种样子。它们原本分散挂在
``BacktestApp`` 上，页面拆成各自的 Mixin 之后就没有共同的归属了——Mixin 里的
``@staticmethod`` 拿不到 ``self``，也就够不着别的 Mixin 上的同类函数。

调用方仍写 ``BacktestApp._format_detail_index(...)``：类里按同名 staticmethod
别名暴露。
"""

import os

import history_selection

from deltalab_ui.constants import BASELINE_STRATEGY_STYLE_KEY


def snapshot_source_label(gui_state):
    source = gui_state.get("source")
    if source == "simulate":
        return f"模拟 · seed {gui_state.get('seed', '—')}"
    if source == "csv":
        path = str(gui_state.get("csv_path", "")).strip()
        return f"CSV · {os.path.basename(path) or '—'}"
    if source == "wind":
        return (
            f"Wind · {gui_state.get('wind_code', '—')} · "
            f"{gui_state.get('wind_start', '—')} 至 "
            f"{gui_state.get('wind_end', '—')} · "
            f"{gui_state.get('wind_bar_size', '日频')}"
        )
    return str(source or "未知来源")


def format_detail_index(value, *, include_time=False):
    """格式化表格索引；日内触发必须保留时分。"""
    if hasattr(value, "strftime"):
        pattern = "%Y-%m-%d %H:%M" if include_time else "%Y-%m-%d"
        try:
            return value.strftime(pattern)
        except (TypeError, ValueError):
            pass
    return str(value)


def format_comparison_value(value, digits=2, *, signed=False,
                             percent=False):
    """稳健格式化对比指标；缺失值统一显示为破折号。"""
    number = history_selection.finite_value(value)
    if number is None:
        return "—"
    if percent:
        number *= 100.0
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    sign = "+" if signed and number > 0 else ""
    suffix = "%" if percent else ""
    return f"{sign}{number:,.{digits}f}{suffix}"


def strategy_style_key(strategy_name, *, fixed_times=None, sigma=None):
    """把策略身份压成跨页稳定的配色键。

    结果对比页的曲线名是用户可改的结果名，策略优选页的是候选名，两者都
    不能直接当配色键。这里只取策略类型及其决定性参数，使同一策略在两页
    取到同一颜色和标记。
    """
    name = str(strategy_name or "").strip()
    if name == "close_to_close":
        return BASELINE_STRATEGY_STYLE_KEY
    if name == "fixed_times":
        times = ",".join(
            part.strip()
            for part in str(fixed_times or "").split(",")
            if part.strip()
        )
        return f"fixed_times:{times or '—'}"
    if name == "hedge_band":
        value = history_selection.finite_value(sigma)
        # σ 是三种带宽输入的共同换算口径；用它当键，绝对/相对输入换算到
        # 同一带宽时也能共享颜色。
        return (
            f"hedge_band:{value:.6g}σ" if value is not None
            else "hedge_band:—")
    return f"other:{name or '—'}"


def history_row_style_key(row):
    """从策略优选排名行取配色键，与快照侧使用同一套命名。"""
    row = dict(row or {})
    strategy_name = str(
        row.get("meta_strategy_name")
        or row.get("strategy_type")
        or "").strip()
    return strategy_style_key(
        strategy_name,
        fixed_times=row.get("meta_fixed_times"),
        sigma=row.get("meta_candidate_sigma"),
    )


def history_baseline_row_label(strategy_label):
    """基线行在策略列里的显示名。

    基线在三种排序口径下取值都恒为 0——增量收益、增量信噪比、以及品种
    池降级时用的逐段改善率——所以它排在哪一位本身就是分界线。这里只标
    明身份，不再拼接分界说明：那句话越写越长，而同一行的 # 列与三个指
    标格已经都显示「基准」了。
    """
    return f"{strategy_label}（基准）"
