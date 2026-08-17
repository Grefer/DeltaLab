# _*_ coding: utf-8 _*_
"""``pricing.windpy_locator`` 的用例：运行时定位本机 Wind 终端的 WindPy。

这批用例全部在 Linux CI 上跑，所以不能碰真实的 Wind 安装、注册表或用户
主目录：Wind 目录用 tmp_path 现造，Windows 专属分支通过 patch
``windpy_locator.os.name`` 触发。
"""
from __future__ import annotations

import os
import sys

import pytest

# 确保可以直接从仓库根导入 pricing 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pricing import wind_data, windpy_locator


@pytest.fixture(autouse=True)
def isolate_process_state(tmp_path, monkeypatch):
    """隔离 locator 会改动的三处进程级状态。

    必须是 autouse：``activate`` 往用户主目录写伪造的 WindPy.pth、往
    ``sys.path`` 插目录、``bootstrap`` 还会往 ``sys.modules`` 塞 WindPy。
    少隔离一处，测试就会污染开发者本机（``~/.deltalab/windpy``）或让后续
    用例拿到上一条留下的假 WindPy。
    """
    shadow = tmp_path / "shadow" / "site-packages"
    monkeypatch.setattr(
        windpy_locator, "_shadow_site_packages", lambda: str(shadow))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "WindPy", raising=False)
    monkeypatch.setattr(windpy_locator, "_DLL_DIR_HANDLES", [])
    return shadow


@pytest.fixture
def without_host_wind(monkeypatch):
    """屏蔽本机真实的 Wind 安装。

    "扫不到就返回 None" 这类用例，在装了 Wind 的机器上（本仓库作者的 mac、
    以及任何真能用 Wind 模式的 Windows 机器）会扫到真货而失败。把平台扫描
    的几张表清空，只留下用例自己传入的 environ / sys_path，结论才与跑测试
    的机器无关。
    """
    for name in ("_SITE_PACKAGES_TEMPLATES", "_MAC_SITE_PACKAGES_TEMPLATES",
                 "_MAC_WIND_DIRS", "_WIND_ROOT_TEMPLATES"):
        monkeypatch.setattr(windpy_locator, name, ())
    monkeypatch.setattr(
        windpy_locator, "_iter_registry_site_packages", lambda: iter(()))


def _make_wind_dir(root, *, with_dll=True, body="w = object()\n"):
    """造一份最小可用的 Wind 目录（WindPy.py [+ WindPy.dll]）。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "WindPy.py").write_text(body, encoding="utf-8")
    if with_dll:
        (root / "WindPy.dll").write_bytes(b"\x00")
    return str(root)


# ---------------------------------------------------------------------------
# 目录校验
# ---------------------------------------------------------------------------

def test_is_windpy_dir_requires_dll_on_windows(tmp_path, monkeypatch):
    """Windows 上只有 .py 没有 .dll 的目录不算数。

    这种半截目录 import 必然死在 ctypes 那一步，当场判否才能让扫描继续走
    到下一个候选（真机上 x86/x64 并存，挑错一个就是这个下场）。
    """
    monkeypatch.setattr(windpy_locator.os, "name", "nt")
    half = _make_wind_dir(tmp_path / "half", with_dll=False)
    full = _make_wind_dir(tmp_path / "full")

    assert windpy_locator.is_windpy_dir(half) is False
    assert windpy_locator.is_windpy_dir(full) is True


def test_is_windpy_dir_rejects_missing_and_empty():
    assert windpy_locator.is_windpy_dir("") is False
    assert windpy_locator.is_windpy_dir(None) is False
    assert windpy_locator.is_windpy_dir("/nonexistent/wind/x64") is False


def test_rejects_other_arch_dir(tmp_path):
    """位数不符的目录要判否，且原因不能说成"文件缺失"。

    这是 32 位 Python 修复 Python 接口后的真实后果：写下的 WindPy.pth 指向
    x86 目录，64 位的 DeltaLab.exe 拿去加载会得到 WinError 193。让它在扫描
    阶段就出局，报错里才能写清是位数问题而不是"没装 Wind"。
    """
    other_arch = "x86" if windpy_locator._ARCH_SUBDIR == "x64" else "x64"
    wrong = _make_wind_dir(tmp_path / "WindNET" / other_arch)

    assert windpy_locator.is_windpy_dir(wrong) is False
    assert "位数不符" in windpy_locator.reject_reason(wrong)


def test_reject_reason_distinguishes_failure_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(windpy_locator.os, "name", "nt")
    empty = tmp_path / "empty"
    empty.mkdir()
    no_dll = _make_wind_dir(tmp_path / "no_dll", with_dll=False)

    assert windpy_locator.reject_reason(str(tmp_path / "gone")) == "目录不存在"
    assert windpy_locator.reject_reason(str(empty)) == "目录在, 但没有 WindPy.py"
    assert windpy_locator.reject_reason(no_dll) == "有 WindPy.py, 但缺 WindPy.dll"
    assert windpy_locator.reject_reason(_make_wind_dir(tmp_path / "ok")) is None


def test_pth_pointing_at_parent_dir_falls_through_to_arch_subdir(tmp_path):
    """.pth 指到 …\\WindNET（位数目录的上一级）时要能下探一层。"""
    wind_net = tmp_path / "Wind.NET.Client" / "WindNET"
    arch_dir = _make_wind_dir(wind_net / windpy_locator._ARCH_SUBDIR)
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "WindPy.pth").write_text(str(wind_net), encoding="utf-8")

    found = windpy_locator.find_windpy_dir(environ={}, sys_path=[str(site)])

    assert found == os.path.normpath(arch_dir)


# ---------------------------------------------------------------------------
# WindPy.pth 解析
# ---------------------------------------------------------------------------

def test_read_pth_dir_handles_gbk_chinese_path(tmp_path):
    """Wind 安装器按系统 ANSI 写 .pth，中文路径必须能按 GBK 兜底读出来。"""
    pth = tmp_path / "WindPy.pth"
    pth.write_bytes("C:\\用户\\Wind\\x64\n".encode("gbk"))

    assert windpy_locator._read_pth_dir(str(pth)) == "C:\\用户\\Wind\\x64"


def test_read_pth_dir_skips_blank_leading_lines(tmp_path):
    pth = tmp_path / "WindPy.pth"
    pth.write_text("\n\n  C:\\Wind\\x64  \n other \n", encoding="utf-8")

    assert windpy_locator._read_pth_dir(str(pth)) == "C:\\Wind\\x64"


def test_read_pth_dir_returns_none_when_missing(tmp_path):
    assert windpy_locator._read_pth_dir(str(tmp_path / "nope.pth")) is None


# ---------------------------------------------------------------------------
# 候选发现
# ---------------------------------------------------------------------------

def test_env_var_wins_over_pth(tmp_path):
    """环境变量是自动扫描失败时的手动出口，必须排在 .pth 之前。"""
    env_dir = _make_wind_dir(tmp_path / "from_env")
    pth_dir = _make_wind_dir(tmp_path / "from_pth")
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "WindPy.pth").write_text(pth_dir, encoding="utf-8")

    found = windpy_locator.find_windpy_dir(
        environ={"DELTALAB_WIND_DIR": env_dir}, sys_path=[str(site)])

    assert found == os.path.normpath(env_dir)


def test_finds_dir_via_site_packages_pth(tmp_path):
    """没有环境变量时，靠 Wind 写下的 WindPy.pth 找到安装目录。"""
    wind_dir = _make_wind_dir(tmp_path / "Wind" / "x64")
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "WindPy.pth").write_text(wind_dir + "\n", encoding="utf-8")

    found = windpy_locator.find_windpy_dir(
        environ={}, sys_path=[str(site)])

    assert found == os.path.normpath(wind_dir)


def test_skips_env_dir_that_fails_validation(tmp_path, monkeypatch):
    """环境变量指到半截目录时不能就此放弃，要继续扫 .pth。"""
    monkeypatch.setattr(windpy_locator.os, "name", "nt")
    broken = _make_wind_dir(tmp_path / "broken", with_dll=False)
    good = _make_wind_dir(tmp_path / "good")
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "WindPy.pth").write_text(good, encoding="utf-8")

    found = windpy_locator.find_windpy_dir(
        environ={"DELTALAB_WIND_DIR": broken}, sys_path=[str(site)])

    assert found == os.path.normpath(good)


def test_find_returns_none_when_nothing_installed(tmp_path, without_host_wind):
    found = windpy_locator.find_windpy_dir(
        environ={}, sys_path=[str(tmp_path / "empty" / "site-packages")])

    assert found is None


def test_wind_install_dirs_match_process_arch():
    """安装目录候选只展开与当前进程同位数的那一档。

    真机上 x86 / x64 两份 WindPy 并存，挑错一边的 WindPy.dll 会以
    WinError 193 失败——那不是"没装 Wind"，用户看到会走错方向。
    """
    joined = "\n".join(windpy_locator._WIND_RELATIVE_DIRS)
    other_arch = "x86" if windpy_locator._ARCH_SUBDIR == "x64" else "x64"

    assert windpy_locator._ARCH_SUBDIR in joined
    assert other_arch not in joined
    assert any("Wind.NET.Client" in d for d in windpy_locator._WIND_RELATIVE_DIRS)


def test_iter_candidates_includes_wind_install_roots():
    candidates = [
        d for _src, d in windpy_locator.iter_candidates(
            environ={}, sys_path=[])
    ]

    assert any("Wind.NET.Client" in d for d in candidates)


def test_describe_search_lists_scanned_paths():
    """报错文本要把扫过的位置摊出来，否则用户只能盲猜。"""
    text = windpy_locator.describe_search(environ={}, sys_path=[])

    assert "Wind" in text
    assert text.count("\n") >= 1


# ---------------------------------------------------------------------------
# 接入当前进程
# ---------------------------------------------------------------------------

def test_activate_puts_shadow_site_packages_first(tmp_path, isolate_process_state):
    """伪造的 WindPy.pth 目录必须叫 site-packages 且排在 sys.path 最前。

    WindPy.py 的 bootstrap 是「遍历 sys.path 找第一个以 site-packages
    结尾的条目，再从里面读 WindPy.pth 拿 DLL 路径」。名字不对它找不到，
    位置不够靠前就会先命中 pyi_rth_windpy 写的那份（指向没有 DLL 的
    _MEIPASS）。
    """
    wind_dir = _make_wind_dir(tmp_path / "Wind" / "x64")

    windpy_locator.activate(wind_dir)

    shadow = str(isolate_process_state)
    assert sys.path[0] == shadow
    assert shadow.endswith("site-packages")
    assert os.path.normpath(wind_dir) in sys.path
    with open(os.path.join(shadow, "WindPy.pth")) as handle:
        assert handle.read().strip() == os.path.normpath(wind_dir)


def test_activate_is_idempotent(tmp_path):
    """重复调用不该把 sys.path 越堆越长（每次连接失败都会重试一遍）。"""
    wind_dir = _make_wind_dir(tmp_path / "Wind" / "x64")

    windpy_locator.activate(wind_dir)
    before = list(sys.path)
    windpy_locator.activate(wind_dir)

    assert sys.path == before


def test_activate_survives_unwritable_shadow_dir(tmp_path, monkeypatch):
    """影子目录写不下去时只降级、不抛：新版 WindPy 靠 __file__ 也能定位 DLL。"""
    monkeypatch.setattr(
        windpy_locator, "_shadow_site_packages",
        lambda: str(tmp_path / "WindPy.py" / "site-packages"))  # 父路径是文件
    (tmp_path / "WindPy.py").write_text("", encoding="utf-8")
    wind_dir = _make_wind_dir(tmp_path / "Wind" / "x64")

    assert windpy_locator.activate(wind_dir) == os.path.normpath(wind_dir)
    assert os.path.normpath(wind_dir) in sys.path


# ---------------------------------------------------------------------------
# bootstrap 端到端
# ---------------------------------------------------------------------------

def test_bootstrap_imports_local_windpy(tmp_path):
    wind_dir = _make_wind_dir(
        tmp_path / "Wind" / "x64",
        body="class _W:\n    tag = 'local'\n\nw = _W()\n")

    module = windpy_locator.bootstrap(
        environ={"DELTALAB_WIND_DIR": wind_dir}, sys_path=[])

    assert module is not None
    assert module.w.tag == "local"
    assert windpy_locator.last_error() is None


def test_bootstrap_returns_none_without_installation(tmp_path, without_host_wind):
    module = windpy_locator.bootstrap(
        environ={}, sys_path=[str(tmp_path / "site-packages")])

    assert module is None


def test_bootstrap_records_import_error(tmp_path):
    """WindPy.py 自身炸掉（真机上多半是 LoadLibrary 失败）时要留下原因。"""
    wind_dir = _make_wind_dir(
        tmp_path / "Wind" / "x64",
        body="raise OSError('WinError 126: 找不到指定的模块')\n")

    module = windpy_locator.bootstrap(
        environ={"DELTALAB_WIND_DIR": wind_dir}, sys_path=[])

    assert module is None
    assert isinstance(windpy_locator.last_error(), OSError)
    assert "WinError 126" in str(windpy_locator.last_error())
    assert "WindPy" not in sys.modules  # 半成品不能留在模块表里


def test_bootstrap_replaces_broken_frozen_windpy(tmp_path):
    """包里那份坏掉的 WindPy 已占位时，改为按文件加载本机的那份。

    冻结进程里 FrozenImporter 排在 sys.meta_path 前面，普通 import 永远
    命中包内那份，光插 sys.path 是没用的。
    """
    import types

    broken = types.ModuleType("WindPy")  # 没有 w 属性 = 半截货
    sys.modules["WindPy"] = broken
    wind_dir = _make_wind_dir(
        tmp_path / "Wind" / "x64",
        body="class _W:\n    tag = 'local'\n\nw = _W()\n")

    module = windpy_locator.bootstrap(
        environ={"DELTALAB_WIND_DIR": wind_dir}, sys_path=[])

    assert module is not None
    assert module.w.tag == "local"


# ---------------------------------------------------------------------------
# 与 _ensure_wind 的接线
# ---------------------------------------------------------------------------

def test_ensure_wind_falls_back_to_locator(monkeypatch):
    """包里没有 WindPy 时，_ensure_wind 要能靠 locator 起来。

    先往 ``sys.modules`` 塞一个没有 ``w`` 的占位模块，模拟"包里没有可用的
    WindPy"。不能只删条目：开发机（本仓库作者的 mac）装了真 WindPy，删掉
    后 ``from WindPy import w`` 会重新加载真模块并去连 Wind 服务，用例要卡
    到连接超时。
    """
    import types

    monkeypatch.setitem(sys.modules, "WindPy", types.ModuleType("WindPy"))
    fake = types.ModuleType("WindPy")
    started = {}

    class _W:
        def isconnected(self):
            return False

        def start(self):
            started["called"] = True
            return types.SimpleNamespace(ErrorCode=0, Data=[])

    fake.w = _W()
    monkeypatch.setattr(windpy_locator, "bootstrap", lambda: fake)

    w = wind_data._ensure_wind()

    assert w is fake.w
    assert started["called"] is True


def test_ensure_wind_error_mentions_scanned_paths_and_env_var(monkeypatch):
    """两条路都失败时，报错要给出扫过的位置和手动指定的办法。"""
    import types

    # 同上：占位模块顶掉真 WindPy，避免在装了 Wind 的机器上真去连服务。
    monkeypatch.setitem(sys.modules, "WindPy", types.ModuleType("WindPy"))
    monkeypatch.setattr(windpy_locator, "bootstrap", lambda: None)
    monkeypatch.setattr(
        windpy_locator, "last_error",
        lambda: OSError("WinError 193: 不是有效的 Win32 应用程序"))

    with pytest.raises(ImportError) as excinfo:
        wind_data._ensure_wind()

    message = str(excinfo.value)
    assert "DELTALAB_WIND_DIR" in message
    assert "已扫描以下位置" in message
    assert "WinError 193" in message
