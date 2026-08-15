# _*_ coding: utf-8 _*_
"""与具体页面无关的 Tk 控件小工具。

两个都是"挂上去就不用再管"的行为增强：让标签的折行宽度跟随实际槽位，以及
给控件挂一个轻量悬浮提示。三个页面都在用，而它们是 ``@staticmethod``——页面
各自成 Mixin 之后，Mixin 里的静态方法拿不到 ``self``，够不着别的 Mixin 上的
同类函数，所以下沉到这里。

调用方仍写 ``BacktestApp._attach_tooltip(...)``：类里按同名 staticmethod
别名暴露，测试也一直是这么调的。
"""

import tkinter as tk

from deltalab_ui.theme import PALETTE, _UI_FONT_FAMILY


def track_wraplength(label, slack=0):
    """让 wraplength 跟随控件实际宽度，而不是写死一个数。

    写死的值必然与真实槽位对不上：基准摘要写的是 980，而它的槽位实测
    1242——文本只要多十来个字就在还剩 262px 的情况下提前折成两行。

    防自激：只有宽度真的变了才重设。设 wraplength 会改变**高度**（一行
    变两行），高度变化会让父容器重排并再次派发 <Configure>；不比宽度就
    会形成「设值→重排→再设值」的闭环。
    """
    last = {"width": None}

    def _apply(event=None):
        width = label.winfo_width()
        if width <= 1 or width == last["width"]:
            return
        last["width"] = width
        try:
            label.configure(wraplength=max(80, width - slack))
        except tk.TclError:
            pass

    label.bind("<Configure>", _apply, add="+")
    label.after_idle(_apply)


def attach_tooltip(widget, text):
    """给控件挂一个轻量悬浮提示。

    用于把过长的辅助信息（例如各周期的最优策略全名）从按钮标题里移
    出去：标题只留周期名，细节悬停可见，按钮宽度就不再被策略名撑爆。
    """
    if not text:
        return
    state = {"tip": None}

    def _show(_event=None):
        if state["tip"] is not None:
            return
        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(
            tip, text=text, bg=PALETTE["text"], fg="#FFFFFF",
            font=(_UI_FONT_FAMILY, 9), padx=8, pady=4, justify="left",
        ).pack()
        x = widget.winfo_rootx() + 12
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        tip.wm_geometry(f"+{x}+{y}")
        state["tip"] = tip

    def _hide(_event=None):
        tip = state.pop("tip", None)
        state["tip"] = None
        if tip is not None:
            tip.destroy()

    widget.bind("<Enter>", _show, add="+")
    widget.bind("<Leave>", _hide, add="+")
    widget.bind("<Destroy>", _hide, add="+")
