# _*_ coding: utf-8 _*_
"""DeltaLab 界面层。

``gui_app.py`` 曾经是一个 11k 行的单文件，这个包按功能域把它拆开。拆分的
两条硬规则：

1. **本包内的模块一律不 import ``gui_app``。** 依赖方向只能是
   ``gui_app`` → ``deltalab_ui``，反向会形成环。跨模块共用的东西（配色、
   度量、显示名映射）必须下沉到 ``theme`` / ``constants``。
2. **本包不做包级 re-export。** ``__init__`` 保持空壳，各模块显式互相
   import。包级 re-export 会让 ``theme`` 的 import 期副作用（选 matplotlib
   后端）在意想不到的时机触发。

``gui_app.py`` 保留为兼容入口：它 re-export 这里的公开名字，测试与外部
调用继续写 ``gui_app.XXX`` 即可。
"""
