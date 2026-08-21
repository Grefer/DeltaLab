# _*_ coding: utf-8 _*_
"""CSV 数据源的导入引导：模板、表头探测与「价格列」候选联动。

这一组守的是**新用户第一次用 CSV 能不能跑通**：模板必须是 ``from_csv`` 直接
读得进的格式（表头写错一个字，用户拿到的就是一份跑不通的样例），表头探测必须
在选文件时就把可用列摆出来，而不是等回测跑到一半从 ``from_csv`` 抛
「列 X 不在 CSV 中」。

不打 ``gui`` 标记：模板与表头是纯函数，控件联动那几条用 ``SimpleNamespace``
假 self 调类级方法，都不需要窗口服务器。
"""
from __future__ import annotations

import io
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from deltalab_ui.constants import OPTION_CLASSES
from deltalab_ui.panel_form import (
    CSV_TEMPLATE_HEADER,
    CSV_TEMPLATE_ROWS,
    FormPanelMixin,
    csv_template_text,
    read_csv_header,
    write_csv_template,
)
from pricing import HedgeBacktest, Option_Vanilla
from pricing.hedge_backtest import _infer_intraday_steps


class _Var:
    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


class _Combo:
    def __init__(self):
        self.values = None

    def configure(self, **kwargs):
        self.values = list(kwargs["values"])


def _fake_app(col="close", combo=None):
    statuses = []
    return SimpleNamespace(
        _csv_path_var=_Var(),
        _csv_col_var=_Var(col),
        _csv_col_combo=combo,
        _set_status=statuses.append,
        statuses=statuses,
    )


def _option(days=4):
    return Option_Vanilla(
        "Vanilla", s0=100.0, sr=[], K=100.0, T=days,
        sigma=0.2, cp=1, r=0.0, q=0.0,
    )


# ---- 模板本身 ----

def test_template_header_matches_from_csv_defaults():
    """第一列是日期、价格列叫 close —— 与 from_csv 不传参时的口径一致。"""
    assert CSV_TEMPLATE_HEADER[0] == "date"
    assert "close" in CSV_TEMPLATE_HEADER
    assert csv_template_text().splitlines()[0] == ",".join(CSV_TEMPLATE_HEADER)


def test_template_parses_as_daily_series(tmp_path):
    path = tmp_path / "行情模板.csv"
    write_csv_template(path)

    frame = pd.read_csv(path, parse_dates=[0], index_col=0)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert list(frame.columns) == list(CSV_TEMPLATE_HEADER[1:])
    assert len(frame) >= 2  # from_csv 的硬下限
    assert frame.index.is_monotonic_increasing
    assert frame["close"].notna().all()
    # 日频模板不该被误判成日内数据（那会把 steps_per_day 推成 >1）。
    assert _infer_intraday_steps(frame.index) == 1


def test_template_dates_skip_weekends(tmp_path):
    """示例日期是交易日序列：断掉周末，用户一眼知道这列不填自然日。"""
    path = tmp_path / "t.csv"
    write_csv_template(path)
    index = pd.read_csv(path, parse_dates=[0], index_col=0).index
    assert set(index.dayofweek) <= {0, 1, 2, 3, 4}


def test_template_covers_the_default_option_tenor():
    """模板长度必须压过界面默认期限，否则「存模板→直接运行」当场报错。

    第一版模板只有 10 行，默认香草期权 22 交易日，点运行收到的是
    「价格序列交易日组不足：期权剩余 22 日，Day 0 后仅观测到 9 个交易日组」——
    引导做到一半反而把人卡在更难懂的报错上。
    """
    tenor = next(
        spec[3]
        for spec in OPTION_CLASSES[next(iter(OPTION_CLASSES))]["params"]
        if spec[0] in ("T_days", "T")
    )
    # Day 0 之后还需要 tenor 个交易日组，故行数至少 tenor + 1。
    assert len(CSV_TEMPLATE_ROWS) >= int(tenor) + 1


def test_template_runs_through_from_csv_with_default_tenor(tmp_path):
    """模板原样就能跑完一次默认参数的回测——用户只需替换数据行。"""
    path = tmp_path / "t.csv"
    write_csv_template(path)

    bt = HedgeBacktest.from_csv(_option(days=22), str(path), multiplier=0)
    result = bt.run()

    assert result is not None
    assert bt._wind_meta["start_date"] == "2024-01-02"
    assert len(bt.prices) >= 23


def test_template_rows_are_self_consistent_ohlc():
    """high/low 包住 open/close：样例本身不能是一眼看去就不像行情的数。"""
    for date, open_, high, low, close, volume in CSV_TEMPLATE_ROWS:
        o, h, l, c = float(open_), float(high), float(low), float(close)
        assert l <= min(o, c) and h >= max(o, c), date
        assert int(volume) > 0


def test_template_is_excel_friendly_utf8_sig(tmp_path):
    path = tmp_path / "t.csv"
    write_csv_template(path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


# ---- 表头探测 ----

def test_read_csv_header_drops_the_date_column():
    """第一列进了 index_col=0，不该出现在价格列候选里。"""
    path = "__header_probe.csv"
    io.open(path, "w", encoding="utf-8-sig", newline="").write(
        "date,open,close\n2024-01-02,1,2\n")
    try:
        assert read_csv_header(path) == ["open", "close"]
    finally:
        os.remove(path)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gbk"])
def test_read_csv_header_handles_common_encodings(tmp_path, encoding):
    path = tmp_path / "h.csv"
    with io.open(path, "w", encoding=encoding, newline="") as handle:
        handle.write("日期,收盘价\n2024-01-02,2.4\n")
    assert read_csv_header(str(path)) == ["收盘价"]


@pytest.mark.parametrize("content", ["", "\n", "date\n2024-01-02\n"])
def test_read_csv_header_returns_empty_when_no_price_column(tmp_path, content):
    """空文件、空行、只有日期一列——都没有可选的价格列。"""
    path = tmp_path / "h.csv"
    path.write_text(content, encoding="utf-8")
    assert read_csv_header(str(path)) == []


def test_read_csv_header_survives_missing_file(tmp_path):
    """读不到不抛：调用方负责提示，不能把选文件这一步炸掉。"""
    assert read_csv_header(str(tmp_path / "不存在.csv")) == []


# ---- 「价格列」候选联动 ----

def test_apply_columns_keeps_a_still_valid_choice():
    combo = _Combo()
    app = _fake_app(col="open", combo=combo)

    assert FormPanelMixin._apply_csv_columns(app, ["open", "close"]) == "open"
    assert app._csv_col_var.get() == "open"
    assert combo.values == ["open", "close"]


def test_apply_columns_prefers_close_when_current_is_gone():
    app = _fake_app(col="收盘", combo=_Combo())
    assert FormPanelMixin._apply_csv_columns(app, ["open", "close"]) == "close"
    assert app._csv_col_var.get() == "close"


def test_apply_columns_falls_back_to_first_column():
    app = _fake_app(col="close", combo=_Combo())
    assert FormPanelMixin._apply_csv_columns(app, ["收盘价", "成交量"]) == "收盘价"
    assert app._csv_col_var.get() == "收盘价"


def test_sync_reports_available_columns(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("date,open,close\n2024-01-02,1,2\n", encoding="utf-8")
    app = _fake_app(col="close", combo=_Combo())

    assert FormPanelMixin._sync_csv_columns(app, str(path)) == "close"
    assert "open" in app.statuses[-1] and "close" in app.statuses[-1]


def test_sync_announces_the_switch_when_column_was_invalid(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("date,open,close\n2024-01-02,1,2\n", encoding="utf-8")
    app = _fake_app(col="不存在的列", combo=_Combo())

    assert FormPanelMixin._sync_csv_columns(app, str(path)) == "close"
    assert "不存在的列" in app.statuses[-1]


def test_sync_leaves_column_alone_when_header_is_unreadable(tmp_path):
    path = tmp_path / "h.csv"
    path.write_text("date\n2024-01-02\n", encoding="utf-8")
    app = _fake_app(col="close", combo=_Combo())

    assert FormPanelMixin._sync_csv_columns(app, str(path)) is None
    assert app._csv_col_var.get() == "close"
    assert app.statuses[-1]


def test_sync_stays_quiet_after_writing_a_template(tmp_path, monkeypatch):
    """模板保存后只报「已生成模板」，不再追一条列名播报刷掉它。"""
    path = tmp_path / "t.csv"
    app = _fake_app(combo=_Combo())
    monkeypatch.setattr(
        "deltalab_ui.panel_form.filedialog.asksaveasfilename",
        lambda **kwargs: str(path))
    monkeypatch.setattr(
        "deltalab_ui.panel_form.messagebox.showinfo", lambda *a, **k: None)

    FormPanelMixin._save_csv_template(app)

    assert path.exists()
    assert app._csv_path_var.get() == str(path)
    assert len(app.statuses) == 1
    assert "模板" in app.statuses[0]


# ---- 两个对话框的取消分支 ----

def test_cancelling_the_save_dialog_writes_nothing(monkeypatch):
    app = _fake_app(combo=_Combo())
    monkeypatch.setattr(
        "deltalab_ui.panel_form.filedialog.asksaveasfilename",
        lambda **kwargs: "")

    FormPanelMixin._save_csv_template(app)

    assert app._csv_path_var.get() == ""
    assert app.statuses == []


def test_cancelling_the_browse_dialog_keeps_the_previous_path(monkeypatch):
    app = _fake_app(combo=_Combo())
    app._csv_path_var.set("旧文件.csv")
    monkeypatch.setattr(
        "deltalab_ui.panel_form.filedialog.askopenfilename",
        lambda **kwargs: "")

    FormPanelMixin._browse_csv(app)

    assert app._csv_path_var.get() == "旧文件.csv"
    assert app.statuses == []


def test_browsing_a_file_syncs_columns(tmp_path, monkeypatch):
    path = tmp_path / "h.csv"
    path.write_text("date,收盘价\n2024-01-02,2.4\n", encoding="utf-8")
    combo = _Combo()
    app = _fake_app(col="close", combo=combo)
    monkeypatch.setattr(
        "deltalab_ui.panel_form.filedialog.askopenfilename",
        lambda **kwargs: str(path))

    FormPanelMixin._browse_csv(app)

    assert app._csv_path_var.get() == str(path)
    assert combo.values == ["收盘价"]
    assert app._csv_col_var.get() == "收盘价"


# ---- 真实窗口里的控件 ----

@pytest.mark.gui
def test_csv_panel_exposes_template_button_and_column_dropdown():
    """CSV 面板上「模板…」和可编辑的「价格列」下拉都在。

    这两个控件是本组逻辑的入口：没有按钮，模板生成永远调不到；「价格列」
    退回普通 Entry 的话 ``_apply_csv_columns`` 灌进去的候选也没处显示。
    """
    import tkinter as tk
    from tkinter import ttk

    try:
        probe = tk.Tk()
    except tk.TclError:
        pytest.skip("无可用显示环境")
    probe.destroy()

    import gui_app as module
    app = module.BacktestApp()
    try:
        app.withdraw()
        app.update_idletasks()

        assert isinstance(app._csv_col_combo, ttk.Combobox)
        # normal 而不是 readonly：表头探测失败时用户还能自己敲列名。
        assert str(app._csv_col_combo.cget("state")) == "normal"
        assert app._csv_col_var.get() == "close"

        buttons = []

        def walk(widget):
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons.append(str(child.cget("text")))
                walk(child)

        walk(app._csv_frame)
        assert "浏览…" in buttons and "模板…" in buttons
    finally:
        app.destroy()
