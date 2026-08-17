# 更新日志

本文件记录 DeltaLab 每个发布版本的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

预编译包在 [Releases](https://github.com/Grefer/DeltaLab/releases) 页下载。

---

## [2.0.0] - 2026-08-16

把 DeltaLab 从「跑一次回测看结果」变成「把多次回测横向比出结论」：新增 **🆚 结果对比** 与 **🎯 策略优选** 两个页面，并把 `gui_app.py` 拆成分层的 `deltalab_ui/` 包。相对 v1.1.0 共 42 个提交、约 +39,500 行。

### 新增

- **🆚 结果对比**：回测后按需保留快照，勾选任意组合即时对照曲线与绝对指标（期末净损益、总成本、最大回撤、触发 / 成交次数、换手额）。切换勾选只读缓存，不重跑回测。快照自动存本机、重开仍在，最多 20 条。页面不替你下判断，只负责说清这次的变量差在哪一项。
- **🎯 策略优选**：用 CSV / Wind 真实历史严格连续回放近周 / 近月 / 近季 / 近半年 / 近年五档，固定以每日收盘 C2C 为基准，按增量收益 / 增量信噪比排名，可把胜出参数一键写回左侧。支持品种入口（`P.DCE` 按 Wind 逐日主力映射）。
- **智能行情粒度**：每日 bar 数由时间索引与交易 session 自动推导，Wind 请求粒度按策略自动选择。
- 分段 bar 级明细的内容寻址缓存；`.env.example` 本地凭据模板。

### 变更

- **`gui_app.py` 拆成 `deltalab_ui/` 包**（15 个模块）。`gui_app.py` 收缩到 1,691 行、保留为兼容入口，外部与测试继续写 `gui_app.XXX` 即可。
- **策略优选周期口径重写**：从「挑若干终点、各自向前回看」改为「从同一截止日向前严格连续回放最近 `L` 日」。
- **移除行情粒度的手工调粗** —— 调粗只会静默漏掉 bar 内触发，让结论偏乐观。
- 指标 `增量性价比` 改名 `增量信噪比`；Wind 回测改为「分析截至日」驱动；历史候选上限 5 → 8；候选带宽的 σ 统一取左侧输入值。
- 表单改用统一布局系统；主题 / 字体 / 布局初始化抽到 `deltalab_ui/theme.py`；按钮与标签视觉层次调整。

### 修复

- **免安装版在装有 Wind 终端的 Windows 上仍报 `No module named 'WindPy'`**：发布包由没有 Wind 终端的 CI 构建，包里不含 WindPy，而运行机上明明装着。现在 `_ensure_wind()` 在 import 失败后会调新增的 `pricing/windpy_locator.py` 扫描本机 Wind 安装（环境变量 → `site-packages` 里的 `WindPy.pth` → 各 Python 安装 → Wind 常见安装目录），接进进程后重试；只取与当前进程同位数的那一档，避免 x86/x64 挑错边的 `WinError 193` 伪装成"没装 Wind"。macOS 的 `/Applications/Wind API.app` 同样覆盖。两条路都失败时的报错会列出扫过的位置，并区分"目录不存在"与"文件齐全但加载失败"；也可用 `DELTALAB_WIND_DIR` 手工指定 Wind 目录。
  > 与 v1.1.0 那条「PyInstaller 打包支持 WindPy」不是同一个问题，是同一条链路的相邻两环：v1.1.0 修的是**包里有 WindPy 但加载不到 DLL**（本地打包场景，靠 runtime hook 在 `_MEIPASS/site-packages/` 伪造 `WindPy.pth`），这次修的是**包里压根没有 WindPy**（CI 打包场景）——那个 hook 只在构建机检测到 WindPy 时才会被打进包，对 CI 包不生效。两者叠起来覆盖两种发布方式；locator 定位 DLL 的手法沿用了 hook 对 `WindPy.py` bootstrap 的分析。
- **内容寻址缓存的键计算缺陷**：不同输入可能命中同一份缓存，会串用别的段的 bar 明细。
- **品种池模式未显式降级排名口径**：跨合约金额被直接相加。
- 策略优选校验的 `UnboundLocalError`，以及排序键混用不同量纲导致的排名错误。
- 回放策略未按分段成功过滤、预热参数未持久化；状态栏打包顺序；历史页分割线定位；headless import 失败。

### 测试与 CI

测试从 23 条（2 个文件）扩到 969 条（13 个文件：440 纯逻辑 + 529 GUI）。CI 在 ubuntu × Python 3.10 / 3.13 上跑全量，GUI 批用 `xvfb-run`。

### 升级说明

- **无需迁移**：结果池与策略优选结果包都是本版新增，v1.x 不存在这两类文件。
- 两类结果包各带 `schema_version`，**版本不符时拒绝载入而不是硬渲染** —— 字段口径改过不止一次，用新代码渲染旧包会静默给出错误结论。
- 三个运行时目录（`data/cache/`、`data/backtest_pool/`、`data/history_results/`）已写进 `.gitignore`：结果池包内含 `csv_path` 全路径与 `wind_code`。
- CI 产出的发布包不内置 WindPy（GitHub runners 上没有 Wind 终端），改为运行时扫描本机 Wind 安装：装了 Wind 终端并设置过 Python 接口的机器可直连，装在非常规位置时用 `DELTALAB_WIND_DIR` 指定。见[使用文档 §1.3](docs/GUI_USAGE.md#13-可选依赖)。

各页面的完整口径、参数含义与常见报错见 [使用文档](docs/GUI_USAGE.md)。

---

## [1.1.0] - 2026-05-22

### 新增

- **雪球期权**（`Option_SNB`）与 `trade_calendar` 交易日模块。
- 累计期权新增 `Opt_Decumulator`（普通累计）与 `Opt_Decumulator_Fix_E`（固赔到期结算累计）两种定价方法。
- PyInstaller 打包支持 WindPy：新增 runtime hook 确保冻结包里 DLL 能正确加载，并给出详细的错误提示。

### 修复

- Windows 打包版因 `charmap` 编码崩溃：spec 中强制 stdout 使用 UTF-8。
- 强制 JavaScript action 使用 Node 24，消除 Node 20 弃用警告。

---

## [1.0.0] - 2026-04-21

首个公开发布。

- 基于 Python / tkinter 的期权 Delta 动态对冲回测 GUI。
- 5 大类期权（香草 / 累计 / 亚式 / 气囊 / 雪球）、3 种数据源（蒙特卡洛 / CSV / Wind API）。
- 6 宫格对冲图表、波动率分析、蒙特卡洛盈亏分布、结构扫描。
- PyInstaller 打包与 GitHub Release 流水线：Windows x86_64 与 macOS arm64 免安装包。
- 修复冻结包下 numpy 2.x 的导入失败（收集 numpy / pandas 子模块与动态库）。

[2.0.0]: https://github.com/Grefer/DeltaLab/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/Grefer/DeltaLab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Grefer/DeltaLab/releases/tag/v1.0.0
