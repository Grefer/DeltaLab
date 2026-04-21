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
