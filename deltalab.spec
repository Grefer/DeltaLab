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
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).resolve()

datas = []
datas += [(str(ROOT / "assets"), "assets")]
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

    # (a) Wind 终端单文件形态: 手工扫 WindPy.__file__ 同目录里的原生库和附属 .py
    _windpy_dir = _os.path.dirname(_os.path.abspath(_windpy_file)) if _windpy_file else None
    if _windpy_dir and _os.path.isdir(_windpy_dir):
        import glob as _glob
        _extra_bins, _extra_datas = 0, 0
        for _path in _glob.glob(_os.path.join(_windpy_dir, "*")):
            if not _os.path.isfile(_path):
                continue
            _name = _os.path.basename(_path)
            _lower = _name.lower()
            if _lower.endswith((".dll", ".pyd", ".so", ".dylib")):
                # 放 bundle 根目录; frozen 运行时 bundle 根在 DLL 搜索路径里,
                # ctypes.CDLL("XxxCom.dll") 这种裸文件名才能找到.
                binaries.append((_path, "."))
                _extra_bins += 1
            elif _lower.endswith(".py") and _lower != "windpy.py":
                # WindPy.py 已由 hiddenimports=["WindPy"] 处理; 同目录其它附属
                # 模块 (WindPyEx / WindCommon 之类) 作为数据文件放 bundle 根,
                # 以便 WindPy.py 内部相对 import 能解析到.
                datas.append((_path, "."))
                _extra_datas += 1
        print(
            f"[deltalab.spec] 从 {_windpy_dir} 手工补齐: "
            f"{_extra_bins} 个原生库 / {_extra_datas} 个附属 .py"
        )

excludes = [
    # 减小体积: 这些在 GUI 里用不到
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook",
    "pytest", "tests",
]

block_cipher = None

a = Analysis(
    ["gui_app.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

_is_windows = sys.platform.startswith("win")
_icon_path = str(ROOT / "assets" / ("deltalab.ico" if _is_windows else "deltalab.png"))

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
        icon=str(ROOT / "assets" / "deltalab.png"),
        bundle_identifier="com.deltalab.app",
        info_plist={
            "CFBundleName": "DeltaLab",
            "CFBundleDisplayName": "DeltaLab",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
