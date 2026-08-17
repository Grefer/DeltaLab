# _*_ coding: utf-8 _*_
"""运行时定位本机 Wind 终端自带的 WindPy，并把它接进当前进程。

为什么需要
----------
发布包由 GitHub Actions 的 windows runner 构建，那台机器上没有 Wind 金融
终端，``deltalab.spec`` 的 WindPy 分支整段跳过 —— 产出的 zip 里根本没有
``WindPy.py``/``WindPy.dll``。用户机器上装了 Wind 终端并"修复"过 Python
接口，模块就在本机磁盘上躺着，只是冻结进程的 ``sys.path`` 里没有它，于是
``import WindPy`` 抛 ``ModuleNotFoundError``。

本模块负责把这一步补上：按优先级扫出本机的 WindPy 目录，把它接进
``sys.path`` / DLL 搜索路径，再伪造一份 ``WindPy.pth`` 满足 WindPy.py 自身
的 bootstrap，最后完成 import。开发态（系统 Python 里 ``import WindPy``
本来就成功）不会走到这里。

发现顺序（先命中先用）
----------------------
1. 环境变量 ``DELTALAB_WIND_DIR`` / ``WINDPY_DIR`` —— 给自动扫描失败的用户留的手动出口
2. ``sys.path`` 上现有 ``site-packages`` 里的 ``WindPy.pth``
3. 本机各 Python 安装（注册表 + 常见路径）的 ``site-packages`` 里的 ``WindPy.pth``
4. Wind 终端的常见安装目录

``WindPy.pth`` 是 Wind 终端"设置 Python 接口"时写下的，第一行就是 Wind 的
安装目录，比逐个猜安装路径可靠得多，所以排在直接扫目录之前。
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import struct
import sys

__all__ = [
    "find_windpy_dir",
    "is_windpy_dir",
    "reject_reason",
    "iter_candidates",
    "activate",
    "bootstrap",
    "describe_search",
    "last_error",
]

# 手动指定 Wind 目录的环境变量；自动扫描全部落空时的兜底出口。
ENV_DIR_KEYS = ("DELTALAB_WIND_DIR", "WINDPY_DIR")

# Wind 安装根下放 WindPy.py 的子目录。x64/x86 两份并存，位数不匹配时
# ctypes 加载 WindPy.dll 会抛 WinError 193，所以只挑与当前进程一致的那份。
_ARCH_SUBDIR = "x64" if struct.calcsize("P") * 8 == 64 else "x86"

# 相对 Wind 安装根的候选层级：新版终端是 Wind.NET.Client\WindNET\x64，
# 老版本 / 绿色版可能直接是 <root>\x64 甚至 <root> 本身。
_WIND_RELATIVE_DIRS = (
    os.path.join("Wind.NET.Client", "WindNET", _ARCH_SUBDIR),
    os.path.join("WindNET", _ARCH_SUBDIR),
    _ARCH_SUBDIR,
    "",
)

# Wind 终端常见安装根。带通配符的走 glob。
_WIND_ROOT_TEMPLATES = (
    r"C:\Wind",
    r"D:\Wind",
    r"E:\Wind",
    r"C:\Software\Wind",
    r"D:\Software\Wind",
    r"{ProgramFiles}\Wind",
    r"{ProgramFiles(x86)}\Wind",
    r"{LOCALAPPDATA}\Wind",
    r"{APPDATA}\Wind",
    r"{USERPROFILE}\Wind",
)

# macOS 的 Wind API 安装位置。mac 版把 WindPy.py 放在 .app 内部，同目录
# 没有 .dll，靠 app 内的 dylib 加载，所以 is_windpy_dir 在非 Windows 上
# 只校验 WindPy.py。
_MAC_WIND_DIRS = (
    "/Applications/Wind API.app/Contents/python",
    "{HOME}/Applications/Wind API.app/Contents/python",
)

# 本机 Python 的 site-packages 常见位置。Windows 上用来找 Wind 写下的
# WindPy.pth；macOS 的 Wind 安装器则是直接往 site-packages 拷 WindPy.py，
# 所以扫描时对每个目录既看 .pth 也看目录自身。
_SITE_PACKAGES_TEMPLATES = (
    r"{LOCALAPPDATA}\Programs\Python\Python3*\Lib\site-packages",
    r"C:\Python3*\Lib\site-packages",
    r"{ProgramFiles}\Python3*\Lib\site-packages",
    r"{USERPROFILE}\anaconda3\Lib\site-packages",
    r"{USERPROFILE}\miniconda3\Lib\site-packages",
    r"{USERPROFILE}\AppData\Local\anaconda3\Lib\site-packages",
    r"C:\ProgramData\Anaconda3\Lib\site-packages",
    r"C:\ProgramData\miniconda3\Lib\site-packages",
)

_MAC_SITE_PACKAGES_TEMPLATES = (
    "/Library/Frameworks/Python.framework/Versions/*/lib/python*/site-packages",
    "{HOME}/Library/Python/*/lib/python/site-packages",
    "/opt/homebrew/lib/python*/site-packages",
    "/usr/local/lib/python*/site-packages",
)

# add_dll_directory 返回的 handle 一旦被回收，目录就从搜索路径里摘掉了，
# 必须模块级持有引用。
_DLL_DIR_HANDLES = []

# 最近一次 bootstrap 的失败原因，供 _ensure_wind 拼进报错文本。
_LAST_ERROR = None


def last_error():
    """返回最近一次 :func:`bootstrap` 的失败异常（没有则 ``None``）。"""
    return _LAST_ERROR


# ---------------------------------------------------------------------------
# 候选目录发现
# ---------------------------------------------------------------------------


def _expand(template, environ):
    """把 ``{VAR}`` 占位符替换成环境变量；缺变量返回 ``None``。"""
    out = template
    start = out.find("{")
    while start >= 0:
        end = out.find("}", start)
        if end < 0:
            break
        name = out[start + 1:end]
        value = environ.get(name)
        if not value:
            return None
        out = out[:start] + value + out[end + 1:]
        start = out.find("{", start + len(value))
    return out


def _iter_glob(pattern):
    """展开通配符；无通配符时原样返回（不做存在性判断，留给校验环节）。"""
    if any(ch in pattern for ch in "*?["):
        yield from sorted(glob.glob(pattern))
    else:
        yield pattern


def reject_reason(directory):
    """返回目录不可用的原因；``None`` 表示这份 WindPy 可用。

    校验点有三个，都是真机上会踩的：

    * 位数：Wind 装出来 x86 / x64 两份并存，而"修复 Python 接口"是按写
      ``.pth`` 的那个 Python 的位数来指的。32 位 Python 指的 x86 目录喂给
      64 位进程，会以 ``WinError 193`` 失败——看起来像"没装 Wind"。
    * ``WindPy.py``：模块本体。
    * ``WindPy.dll``：Windows 上缺它必然死在 ctypes 那步。macOS 版把动态库
      放在 ``Wind API.app`` 内部，同目录没有 dll，所以不查。
    """
    if not directory:
        return "路径为空"
    other_arch = "x86" if _ARCH_SUBDIR == "x64" else "x64"
    try:
        # 先判存在再判位数：不存在的路径说"位数不符"只会误导。
        if not os.path.isdir(directory):
            return "目录不存在"
        if os.path.basename(directory.rstrip("\\/")).lower() == other_arch:
            return f"是 {other_arch} 版, 与当前 {_ARCH_SUBDIR} 进程位数不符"
        if not os.path.isfile(os.path.join(directory, "WindPy.py")):
            return "目录在, 但没有 WindPy.py"
        if os.name == "nt" and not os.path.isfile(
                os.path.join(directory, "WindPy.dll")):
            return "有 WindPy.py, 但缺 WindPy.dll"
    except OSError as exc:
        return f"无法访问: {exc}"
    return None


def is_windpy_dir(directory):
    """目录是否是一份可用的 WindPy。"""
    return reject_reason(directory) is None


def _read_pth_dir(pth_path):
    """读 ``WindPy.pth`` 的第一行非空内容（即 Wind 安装目录）。

    编码逐个试：Wind 安装器按系统 ANSI（简中机器上是 GBK）写，中文用户名
    路径下用 UTF-8 读会直接 UnicodeDecodeError。
    """
    for encoding in ("utf-8-sig", "gbk", "latin-1"):
        try:
            with open(pth_path, "r", encoding=encoding) as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        return text
            return None
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _iter_registry_site_packages():
    """从注册表里的 Python 安装读出 ``Lib\\site-packages``。"""
    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:
        return
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            core = winreg.OpenKey(root, r"SOFTWARE\Python\PythonCore")
        except OSError:
            continue
        try:
            index = 0
            while True:
                try:
                    version = winreg.EnumKey(core, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(
                            core, version + r"\InstallPath") as key:
                        install, _ = winreg.QueryValueEx(key, "")
                except OSError:
                    continue
                if install:
                    yield os.path.join(
                        str(install), "Lib", "site-packages")
        finally:
            core.Close()


def _iter_site_packages(environ, sys_path):
    """列出可能藏着 ``WindPy.pth`` 的 site-packages 目录。"""
    seen = set()

    def _emit(path):
        if not path:
            return
        key = os.path.normcase(os.path.normpath(path))
        if key in seen:
            return
        seen.add(key)
        return path

    # 当前进程 sys.path 上的（开发态直接命中；冻结态一般没有）
    for entry in sys_path:
        if entry and entry.rstrip("\\/").endswith("site-packages"):
            path = _emit(entry)
            if path:
                yield path

    for path in _iter_registry_site_packages():
        path = _emit(path)
        if path:
            yield path

    templates = _SITE_PACKAGES_TEMPLATES
    if sys.platform == "darwin":
        # 当前平台的模板排前面：另一边的路径在本机必然不存在，让它们抢占
        # describe_search 的展示条数只会让诊断文本更难读。
        templates = _MAC_SITE_PACKAGES_TEMPLATES + templates
    for template in templates:
        expanded = _expand(template, environ)
        if not expanded:
            continue
        for path in _iter_glob(expanded):
            path = _emit(path)
            if path:
                yield path


def iter_candidates(environ=None, sys_path=None):
    """产出 ``(来源说明, 候选目录)``，按优先级排列，不做存在性过滤。

    报错文本要把扫过哪些位置摊给用户看，所以这里保留全部候选（含不存在
    的），过滤交给 :func:`find_windpy_dir`。
    """
    environ = os.environ if environ is None else environ
    sys_path = sys.path if sys_path is None else sys_path

    for key in ENV_DIR_KEYS:
        raw = (environ.get(key) or "").strip().strip('"')
        if raw:
            yield (f"环境变量 {key}", raw)

    for site_dir in _iter_site_packages(environ, sys_path):
        pth = os.path.join(site_dir, "WindPy.pth")
        if os.path.isfile(pth):
            target = _read_pth_dir(pth)
            if target:
                yield (f"WindPy.pth ({pth})", target)
                # 有的 .pth 指到位数目录的上一级（…\WindNET），下探一层。
                yield (f"WindPy.pth ({pth})",
                       os.path.join(target, _ARCH_SUBDIR))
        # pip 装的 WindPy、以及 macOS 版 Wind 直接往 site-packages 拷的
        # WindPy.py，都在目录自身里。
        yield ("site-packages", site_dir)

    if sys.platform == "darwin":
        for template in _MAC_WIND_DIRS:
            expanded = _expand(template, environ)
            if expanded:
                yield ("Wind API.app", expanded)

    # Windows 安装目录这一段不按平台设防：非 Windows 上这些路径根本不存在，
    # 校验环节自然跳过，多写一层平台分支只会让候选顺序更难推理。
    for template in _WIND_ROOT_TEMPLATES:
        expanded = _expand(template, environ)
        if not expanded:
            continue
        for root in _iter_glob(expanded):
            for relative in _WIND_RELATIVE_DIRS:
                yield ("常见安装目录", os.path.join(root, relative) if relative else root)


def find_windpy_dir(environ=None, sys_path=None):
    """返回第一个通过 :func:`is_windpy_dir` 校验的候选目录，找不到返回 ``None``。"""
    for _source, directory in iter_candidates(environ, sys_path):
        if is_windpy_dir(directory):
            return os.path.normpath(directory)
    return None


def describe_search(environ=None, sys_path=None, limit=12):
    """生成"扫了哪些位置"的诊断文本，拼进 ImportError 给用户看。

    存在却没通过校验的目录（典型是 x86/x64 挑错了一边、或 Wind 装了但
    Python 接口没修复）最值得用户看，所以排在压根不存在的路径前面，
    ``limit`` 截断时优先保住它们。
    """
    existing, missing = [], []
    seen = set()
    for source, directory in iter_candidates(environ, sys_path):
        shown = os.path.normpath(directory)
        key = os.path.normcase(shown)
        if key in seen:
            continue
        seen.add(key)
        reason = reject_reason(directory)
        if reason is None:
            # 目录合格却走到这条报错里 = 文件都在但 import 炸了（Windows 上
            # 多半是 LoadLibrary 失败）。标出来，别让用户去查根本不存在的
            # "文件缺失"。
            existing.insert(0, f"    - {shown}  [{source}] 文件齐全, 但加载失败")
        elif reason == "目录不存在":
            missing.append(f"    - {shown}  [{source}]")
        else:
            existing.append(f"    - {shown}  [{source}] {reason}")

    lines = existing + missing
    if len(lines) > limit:
        lines = lines[:limit] + [f"    - ...(另有 {len(lines) - limit} 处候选未列出)"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 接入当前进程
# ---------------------------------------------------------------------------


def _shadow_site_packages():
    """返回用于放伪造 ``WindPy.pth`` 的目录（末级必须叫 site-packages）。

    WindPy.py 在 import 阶段遍历 ``sys.path`` 找**以 site-packages 结尾**的
    条目，再从里面读 ``WindPy.pth`` 拿 DLL 路径。冻结进程没有 site-packages，
    这里在用户目录下造一个顶上。写用户目录而不是 ``_MEIPASS``：发布包可能被
    解压到只读位置（Program Files）。
    """
    return os.path.join(
        os.path.expanduser("~"), ".deltalab", "windpy", "site-packages")


def activate(directory):
    """把 Wind 目录接进当前进程：sys.path、DLL 搜索路径、伪造 WindPy.pth。

    幂等：重复调用只是重复写同样的内容，不会叠加 sys.path 条目。
    """
    directory = os.path.normpath(directory)

    # 1. 让 import 能找到 WindPy.py 及其同目录附属模块
    if directory not in sys.path:
        sys.path.insert(0, directory)

    # 2. WindPy.dll 依赖同目录的其它原生库；Python 3.8+ 起 ctypes 不再看
    #    PATH，必须显式登记 DLL 目录。
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))
        except OSError:
            pass

    # 3. 伪造 WindPy.pth 供 WindPy.py 自身的 bootstrap 读取。必须插在
    #    sys.path 最前面：冻结包里可能已有 pyi_rth_windpy 写的那份（指向
    #    _MEIPASS），那边没有 DLL，让它先命中就白搭了。
    shadow = _shadow_site_packages()
    try:
        os.makedirs(shadow, exist_ok=True)
        # 不指定 encoding，与 WindPy.py 里无编码的 open(sitepath) 对齐，
        # 中文路径下才不会一边 UTF-8 写一边 GBK 读。
        with open(os.path.join(shadow, "WindPy.pth"), "w") as handle:
            handle.write(directory)
        if sys.path and sys.path[0] == shadow:
            pass
        else:
            if shadow in sys.path:
                sys.path.remove(shadow)
            sys.path.insert(0, shadow)
    except OSError:
        # 写不了影子目录不致命：新版 WindPy.py 用 __file__ 同目录定位 DLL，
        # 光靠第 1、2 步也可能成功。让 import 去决定成败。
        pass

    # 进程启动后才往 sys.path 里加目录，必须让 import 机制重扫，否则可能
    # 命中启动时缓存的"这条路径下没有 WindPy"。
    importlib.invalidate_caches()

    return directory


def _import_from_file(directory):
    """绕开冻结包里的模块图，直接按文件加载本机 WindPy.py。

    冻结进程里 PyInstaller 的 FrozenImporter 排在 sys.meta_path 前面，如果
    发布包里带了一份坏掉的 WindPy（构建机能 import、运行机缺 DLL），普通
    ``import WindPy`` 会一直命中那份，sys.path 插多少遍都没用。
    """
    path = os.path.join(directory, "WindPy.py")
    spec = importlib.util.spec_from_file_location("WindPy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 构造 WindPy 模块规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules["WindPy"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("WindPy", None)
        raise
    return module


def bootstrap(environ=None, sys_path=None):
    """扫描 + 接入 + import，成功返回 WindPy 模块，失败返回 ``None``。

    失败原因记在 :func:`last_error` 里，由调用方决定怎么呈现。
    """
    global _LAST_ERROR
    _LAST_ERROR = None

    directory = find_windpy_dir(environ, sys_path)
    if directory is None:
        return None

    activate(directory)

    # 已经 import 过一次失败的残件要清掉，否则 import 会拿到半成品。
    stale = sys.modules.get("WindPy")
    if stale is not None and not hasattr(stale, "w"):
        sys.modules.pop("WindPy", None)

    try:
        module = importlib.import_module("WindPy")
        if hasattr(module, "w"):
            return module
        # 命中了没有 w 的模块（冻结包里的坏件），退回按文件加载
        sys.modules.pop("WindPy", None)
    except Exception as exc:  # noqa: BLE001 - 记录后回退到按文件加载
        _LAST_ERROR = exc

    try:
        return _import_from_file(directory)
    except Exception as exc:  # noqa: BLE001 - 交给调用方拼报错
        _LAST_ERROR = exc
        return None
