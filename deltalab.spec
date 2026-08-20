# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for DeltaLab.

Build (run from repo root):
    pyinstaller --noconfirm deltalab.spec

Outputs:
    dist/DeltaLab/              (Windows / Linux onedir)
    dist/DeltaLab.app           (macOS .app bundle, BUNDLE only runs on macOS)
"""

import sys

# 让 spec 内的中文 print 在任意平台都可输出：Windows runner 默认 stdout 用
# cp1252/charmap，遇中文会抛 UnicodeEncodeError 直接中断构建。强制切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).resolve()


def _resolve_version():
    """本次构建的版本号，形如 ``2.0.0``。

    优先级：``DELTALAB_VERSION`` 环境变量（release.yml 从 tag 注入）→ 本地
    ``git describe`` 的最近 tag → ``0.0.0``。

    此前 CFBundleShortVersionString 是写死的 "1.0.0"，而仓库当时已经打到
    v2.0.0：用户在「关于本机 → 系统报告」或 Finder 简介里看到的版本，和他
    下载的那个包对不上，报 bug 时给出的版本号也就没有意义。
    """
    import os
    import re
    import subprocess

    raw = os.environ.get("DELTALAB_VERSION", "").strip()
    if not raw:
        try:
            raw = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=str(ROOT), capture_output=True, text=True,
                timeout=10, check=True).stdout.strip()
        except Exception:
            raw = ""
    raw = raw.lstrip("vV")
    # CFBundleShortVersionString 只接受 1~3 段数字；tag 上的后缀（-rc1 等）
    # 要削掉，否则 Finder 与 codesign 会拒绝整个 plist。
    match = re.match(r"^\d+(\.\d+){0,2}", raw)
    return match.group(0) if match else "0.0.0"


VERSION = _resolve_version()
print(f"[deltalab.spec] build version = {VERSION}")

datas = []
datas += [(str(ROOT / "assets"), "assets")]
# 离线交易日历, 供雪球(Option_SNB)自然日计息/日期换算使用, 避免运行时依赖 Wind/akshare.
if (ROOT / "data" / "tradingday.csv").exists():
    datas += [(str(ROOT / "data" / "tradingday.csv"), "data")]
datas += collect_data_files("matplotlib")
# numpy 2.x 把 numpy.core 改名为 numpy._core, 有些子模块 (如 _exceptions) 需要
# 显式通过 data 文件形式带上元数据, 避免运行时 "No module named numpy._core._exceptions".
datas += collect_data_files("numpy")

binaries = []
# 把 numpy/scipy/pandas/pyarrow 的 C 扩展 (.pyd/.so) 全部收齐, 防止漏掉某个
# _multiarray_umath / _exceptions 之类的动态库.
for _pkg in ("numpy", "scipy", "pandas", "pyarrow"):
    binaries += collect_dynamic_libs(_pkg)

hiddenimports = []
# numpy 2.x 的私有子模块在模块图里常被遗漏, 强制全收.
hiddenimports += collect_submodules("numpy")
# scipy 有大量子模块通过字符串间接导入, 一次性收齐避免运行时 ImportError.
hiddenimports += collect_submodules("scipy")
# pandas / pyarrow 的 IO 后端
hiddenimports += collect_submodules("pandas")
hiddenimports += ["pyarrow", "pyarrow.parquet", "pyarrow.vendored.version"]
# 项目内部的定价子模块 (通过字符串/懒加载使用)
hiddenimports += collect_submodules("pricing")

# ---------------------------------------------------------------------------
# WindPy: 可选打包
# ---------------------------------------------------------------------------
# 只有**构建机**上能 import WindPy 时才把它打进发布包. 两种形态都兼容:
#   (a) Wind 终端自带的单文件 WindPy.py (例: C:\Software\Wind\x64\WindPy.py)
#       —— 同目录还有 *.dll / *.pyd / 附属 .py, 需要手工扫
#   (b) pip install WindPy 的包形态 —— collect_* 可以处理
# CI (GitHub runners): 没装 Wind 终端 -> 自动跳过, Wind 模式仍报 "未安装 WindPy".
import os as _os
_has_windpy = False
_windpy_file = None
try:
    import WindPy  # noqa: F401
    _has_windpy = True
    _windpy_file = getattr(WindPy, "__file__", None)
except Exception as _windpy_err:
    print(f"[deltalab.spec] WindPy 不可用, 跳过打包 ({_windpy_err!r})")

if _has_windpy:
    print(f"[deltalab.spec] WindPy 已安装 ({_windpy_file}), 打入发布包")
    hiddenimports += ["WindPy"]

    # (b) pip 包形态: 即使找不到也不算失败
    try:
        _sub = collect_submodules("WindPy")
        hiddenimports += _sub
        if _sub:
            print(f"[deltalab.spec] collect_submodules(WindPy) -> {len(_sub)} 项")
    except Exception as _e:
        print(f"[deltalab.spec] collect_submodules(WindPy) 跳过: {_e!r}")
    try:
        _bins = collect_dynamic_libs("WindPy")
        binaries += _bins
        if _bins:
            print(f"[deltalab.spec] collect_dynamic_libs(WindPy) -> {len(_bins)} 项")
    except Exception as _e:
        print(f"[deltalab.spec] collect_dynamic_libs(WindPy) 跳过: {_e!r}")
    try:
        _data = collect_data_files("WindPy")
        datas += _data
        if _data:
            print(f"[deltalab.spec] collect_data_files(WindPy) -> {len(_data)} 项")
    except Exception as _e:
        print(f"[deltalab.spec] collect_data_files(WindPy) 跳过: {_e!r}")

    # (a) Wind 终端单文件形态: 手工扫 WindPy.__file__ 同目录里的所有依赖
    # 除 .exe (不要带 Wind 客户端本体) 外全部打入; 包括:
    #   - *.dll / *.pyd : 原生库 -> binaries (放 bundle 根以便 DLL 搜索)
    #   - WindPy.py     : 由 hiddenimports + module_collection_mode 处理, 跳过
    #   - 其它 *.py     : WindPyEx / WindCommon 等附属模块 -> datas
    #   - *.pth         : Wind 自定义路径配置, WindPy.py 初始化会读 -> datas
    #   - *.ini/.dat/.cfg/.txt/.json/.xml 等杂项 -> 统一 datas
    _windpy_dir = _os.path.dirname(_os.path.abspath(_windpy_file)) if _windpy_file else None
    if _windpy_dir and _os.path.isdir(_windpy_dir):
        import glob as _glob
        _extra_bins, _extra_datas = 0, 0
        for _path in _glob.glob(_os.path.join(_windpy_dir, "*")):
            if not _os.path.isfile(_path):
                continue
            _name = _os.path.basename(_path)
            _lower = _name.lower()
            if _lower.endswith(".exe"):
                continue  # 跳过 Wind 客户端可执行文件, 不该打进发布包
            if _lower == "windpy.py":
                continue  # 由 hiddenimports + module_collection_mode 处理
            if _lower.endswith((".dll", ".pyd", ".so", ".dylib")):
                # 放 bundle 根目录; frozen 运行时 bundle 根在 DLL 搜索路径里,
                # ctypes.CDLL("XxxCom.dll") 这种裸文件名才能找到.
                binaries.append((_path, "."))
                _extra_bins += 1
            else:
                # 其它 .py / .pth / .ini / .dat / .cfg / .txt / .json 等全部
                # 作为数据文件放 bundle 根. 和 WindPy.py 同目录, 相对路径解析
                # (os.path.dirname(__file__)) 能找到.
                datas.append((_path, "."))
                _extra_datas += 1
        print(
            f"[deltalab.spec] 从 {_windpy_dir} 手工补齐: "
            f"{_extra_bins} 个原生库 / {_extra_datas} 个附属资源"
        )

    # 强制 WindPy.py 以 "松散 .py + PYZ .pyc" 两份形式存在:
    # 默认 PyInstaller 只把 .py 编译进 PYZ 归档, 导致 WindPy.__file__ 指向的
    # _MEIPASS/WindPy.py 在磁盘上并不存在, 内部 open(os.path.join(
    # os.path.dirname(__file__), "WindPy.pth")) 会拿到一个不存在的路径.
    # 用 "pyz+py" 让 PyInstaller 在 _MEIPASS 里同时放一份真实 WindPy.py,
    # 相对路径解析与 Wind 安装目录里的行为一致.
    _module_collection_mode = {"WindPy": "pyz+py"}
else:
    _module_collection_mode = {}

excludes = [
    # 减小体积: 这些在 GUI 里用不到
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook",
    "pytest", "tests",
]

block_cipher = None

# WindPy 专用 runtime hook: 在 import WindPy 之前在 _MEIPASS/site-packages/ 伪造
# 一份 WindPy.pth, 让 WindPy.py 的 bootstrap 能定位到 WindPy.dll.
# 仅当构建机检测到 WindPy 时才加载; 否则 runtime hook 本身是 no-op, 加不加无所谓,
# 但为免 CI 打到不存在的文件, 还是按需加入.
_runtime_hooks = []
if _has_windpy:
    _rthook = str(ROOT / "pyi_rth_windpy.py")
    if _os.path.isfile(_rthook):
        _runtime_hooks.append(_rthook)
    else:
        print(f"[deltalab.spec] 警告: {_rthook} 不存在, WindPy 可能无法在运行时加载 DLL")

a = Analysis(
    ["gui_app.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=_runtime_hooks,
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    module_collection_mode=_module_collection_mode,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_is_windows = sys.platform.startswith("win")
_is_macos = sys.platform == "darwin"
_icon_path = str(ROOT / "assets" / ("deltalab.ico" if _is_windows else "deltalab.icns" if _is_macos else "deltalab.png"))

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeltaLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI 应用, 不弹出控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DeltaLab",
)

# macOS: 额外产出一个 .app bundle, 让用户可以直接双击运行.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="DeltaLab.app",
        icon=str(ROOT / "assets" / "deltalab.icns"),
        bundle_identifier="com.deltalab.app",
        info_plist={
            "CFBundleName": "DeltaLab",
            "CFBundleDisplayName": "DeltaLab",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
