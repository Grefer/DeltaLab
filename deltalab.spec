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
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve()

datas = []
datas += [(str(ROOT / "assets"), "assets")]
datas += collect_data_files("matplotlib")

hiddenimports = []
# scipy 有大量子模块通过字符串间接导入, 一次性收齐避免运行时 ImportError.
hiddenimports += collect_submodules("scipy")
# pandas / pyarrow 的 IO 后端
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
    binaries=[],
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
