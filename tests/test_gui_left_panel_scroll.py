"""左侧参数面板的滚轮行为。

回归背景：面板原来用 ``<Enter>``/``<Leave>`` 动态挂载 ``bind_all`` 滚轮。指针
只要从 Canvas 挪到面板里任意子控件（LabelFrame / Label / Entry / Combobox…）
上，Canvas 就收到 ``<Leave>`` 把滚轮解绑——而参数区表面几乎全被子控件盖住，
于是「鼠标停在参数上滚轮没反应，只有压着右侧滚动条才滚得动」。

这里不模拟真实指针（无头环境下的坐标命中不稳），而是替换 ``winfo_containing``
直接指定指针下的控件，专门盯住「按控件归属判断是否滚面板」这段逻辑。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gui_app

pytestmark = pytest.mark.gui


def _make_app():
    import tkinter as tk
    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    app = gui_app.BacktestApp()
    app.withdraw()
    app.update_idletasks()
    # 无头/未映射时控件高度是 1，滚动量会退化成 0。显式给一个滚动步长和
    # 远大于视口的 scrollregion，让 yview_scroll 的效果可判定。
    app._left_canvas.configure(yscrollincrement=10, scrollregion=(0, 0, 200, 4000))
    return app


def _deepest_left_child(app):
    """取左侧面板里最深的一个子控件，模拟指针停在某个参数控件上。"""
    deepest, best_depth = None, -1

    def walk(widget, depth):
        nonlocal deepest, best_depth
        if depth > best_depth:
            deepest, best_depth = widget, depth
        for child in widget.winfo_children():
            walk(child, depth + 1)

    walk(app._left_inner, 0)
    assert best_depth >= 2, "左侧面板没有嵌套子控件，这个回归就测不到点子上"
    return deepest


def _wheel_down():
    # 同时带上 num 与 delta：Linux 走 <Button-5>，Windows/macOS 走 <MouseWheel>，
    # 两条路径下都是「向下滚一格」。
    return SimpleNamespace(num=5, delta=-120, x_root=0, y_root=0)


def test_wheel_over_a_nested_child_scrolls_the_left_panel():
    app = _make_app()
    try:
        child = _deepest_left_child(app)
        app.winfo_containing = lambda _x, _y: child

        before = app._left_canvas.yview()[0]
        assert app._left_wheel_handler(_wheel_down()) == "break"
        assert app._left_canvas.yview()[0] > before
    finally:
        app.destroy()


def test_wheel_outside_the_left_panel_is_left_to_the_right_side():
    """指针在右侧结果区时不许劫持事件，否则图表/表格自己的滚动就废了。"""
    app = _make_app()
    try:
        app.winfo_containing = lambda _x, _y: app._nb

        before = app._left_canvas.yview()[0]
        assert app._left_wheel_handler(_wheel_down()) is None
        assert app._left_canvas.yview()[0] == before
    finally:
        app.destroy()


def test_wheel_binding_survives_leaving_and_reentering_the_canvas():
    """<Leave> 不能再把滚轮绑定摘掉——这正是原来的故障点。"""
    app = _make_app()
    try:
        child = _deepest_left_child(app)
        app.winfo_containing = lambda _x, _y: child

        app.event_generate("<Leave>", when="now")
        app._left_canvas.event_generate("<Leave>", when="now")
        app.update_idletasks()

        before = app._left_canvas.yview()[0]
        assert app._left_wheel_handler(_wheel_down()) == "break"
        assert app._left_canvas.yview()[0] > before

        sequences = ("<Button-4>", "<Button-5>") if gui_app._SYSTEM == "Linux" \
            else ("<MouseWheel>",)
        for sequence in sequences:
            assert app.bind_all(sequence), f"{sequence} 的全局绑定被摘掉了"
    finally:
        app.destroy()


def test_wheel_no_longer_changes_combobox_values():
    """滚轮悬停在下拉框上只滚面板，不许顺手把「大类」改掉。"""
    app = _make_app()
    try:
        for widget_class in ("TCombobox", "TSpinbox"):
            script = app.bind_class(widget_class, "<MouseWheel>")
            assert "Scroll" not in script and "MouseWheel" not in script, (
                f"{widget_class} 仍绑着 Tk 自带的滚轮改值行为: {script}")
    finally:
        app.destroy()
