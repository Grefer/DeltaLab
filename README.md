# DeltaLab

![DeltaLab](assets/banner.png)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Grefer/DeltaLab)](https://github.com/Grefer/DeltaLab/commits/master)
[![Issues](https://img.shields.io/github/issues/Grefer/DeltaLab)](https://github.com/Grefer/DeltaLab/issues)

> 基于 Python / tkinter 的期权 Delta 动态对冲回测框架，支持多种奇异期权、多数据源接入、日内多频率调仓与蒙特卡洛分析。

---

## ✨ 功能特性

- **5 大类期权**：香草 (Vanilla)、累计 (Decumulator)、亚式 (Asian)、气囊 (Airbag)、雪球 (Snowball)，共 14 种子类型
- **3 种数据源**：蒙特卡洛模拟 / CSV 历史行情 / Wind API
- **3 种对冲触发方式**：每日收盘 / 每日固定时刻 / 固定价格间隔（绝对价格、相对价格或日波动 σ 三种执行口径，可在参考点互相换算）
- **日内多频率**：模拟路径可选择每日采样 bar 数；CSV / Wind 真实行情按时间索引与交易 session 自动推导每日 bar 数
- **回测结果池对比**：逐次回测后按需保留结果，换策略或参数继续运行，再在独立对比页勾选快照即时比较
- **独立历史择优**：在专属页面配置候选策略与参数，使用 CSV / Wind 真实历史给出近周、月、季度、年的择优结果与样本诊断，并可把推荐参数带回单次回测验证
- **独立路径采样**：独立 `mc_seed` 多路径采样，避免样本相关性过高
- **完整可视化**：6 宫格对冲图表、波动率分析、蒙特卡洛盈亏分布、结构扫描，支持每日明细导出
- **真实行情缩放**：CSV / Wind 模式下，以首日价为锚点等比例自动缩放期权参数（行权价、障碍价、赔付金额等），保持结构的相对一致性

## 🗺️ 工作流概览

![DeltaLab 工作流](assets/workflow.png)

## 🚀 快速开始

### 环境要求

- **Python 3.10+**（使用了 `match/case` 语法）
- 推荐 3.11 或更高版本

### 安装依赖

```bash
pip install -r requirements.txt
```

> Wind 数据源为可选依赖，需安装 Wind 终端 + Python 插件（`WindPy`）。模拟和 CSV 模式无需 Wind。

### 启动 GUI

```bash
python gui_app.py
```

窗口默认大小 1600×1000，最小 1200×720。启动后会自动加载 [assets/deltalab.ico](assets/deltalab.ico)（Windows）或 [assets/deltalab.png](assets/deltalab.png)（macOS / Linux）作为窗口图标。

### 运行回测

1. **期权类型** → 选 `香草期权 (Vanilla)`
2. **期权参数** → 保留默认（ATM Call, 22 天, σ=0.18）
3. **回测设置** → 数据来源选 `模拟`，模拟路径数改为 `500`
4. 点击 **▶ 运行回测**
5. 点击 **＋ 保留当前结果到对比**，修改对冲策略或参数后再次运行并保留
6. 打开 **⚖ 回测结果对比**，勾选快照即时查看累计净 PnL、排名和差异
7. 查看 `回测摘要` / `对冲图表` / `波动率分析` / `盈亏分布` 等标签页

### 📦 下载预编译版本

Windows / macOS (Apple Silicon) 用户可直接从 [Releases](https://github.com/Grefer/DeltaLab/releases) 下载免安装包（无需 Python 环境）：

- `DeltaLab-vX.Y.Z-windows-x86_64.zip` — 解压后双击 `DeltaLab.exe`
- `DeltaLab-vX.Y.Z-macos-arm64.zip` — Apple Silicon (M 系列)

> macOS 首次打开若提示"未知开发者"，请在 `访达 → 应用程序` 中 **右键 → 打开**，或在 `系统设置 → 隐私与安全性` 中允许。
>
> Intel Mac 用户请从源码运行（见上方"启动 GUI"），仅需 `pip install -r requirements.txt && python gui_app.py`。

## 📁 项目结构

```
DeltaLab/
├── gui_app.py        # GUI 入口 (tkinter + matplotlib)
├── pricing/          # 核心定价与回测引擎 (期权类 / MC / HedgeBacktest)
├── tests/            # 测试
├── data/             # 交易日历与 Wind 数据缓存 (运行时生成)
├── assets/           # 图标 / banner / 工作流示意图
├── tools/            # 资源生成脚本 (make_icon / make_banner / make_workflow)
└── docs/             # 深度文档 (GUI_USAGE.md)
```

文件级结构、各 `Option_*` 的定价方式、引擎内部接口等细节见 [docs/GUI_USAGE.md](docs/GUI_USAGE.md)。

## 📊 支持的期权类型

可以通过点击 **📊 绘制结构图** 查看期权结构说明与 Greeks 扫描曲线。

| 大类 | 子类型数 | 定价方式 |
|---|---|---|
| 香草期权 (Vanilla) | 1（欧式） | Black-Scholes 封闭解 |
| 累计期权 (Decumulator) | 9（回归 / 增强 / 固赔 等系列） | 蒙特卡洛 |
| 亚式期权 (Asian) | 2（亚式, 增强亚式） | 蒙特卡洛 |
| 气囊期权 (Airbag) | 1（气囊） | 蒙特卡洛 |
| 雪球期权 (Snowball) | 1（雪球 / 反雪球） | 蒙特卡洛 |

子类型完整清单与 payoff 公式见 [docs/GUI_USAGE.md §4.1](docs/GUI_USAGE.md)。

## 🔧 技术栈

**Python 3.10+** · **tkinter + ttk** (GUI) · **matplotlib** (图表) · **numpy / scipy** (定价) · **pandas** (数据) · **WindPy** (实时行情, 可选) · **ThreadPoolExecutor** (多路径 MC)

## 对冲模式与策略比较

回测引擎支持每日收盘 `CloseToCloseStrategy`、每天多个固定时刻
`FixedTimeStrategy(["11:30", "15:00"])`，以及统一带宽策略。
`HedgeBandStrategy` 可选择绝对价格、相对百分比或动态 σ 三种阈值；
例如 `HedgeBandStrategy("absolute", threshold=50)`、
`HedgeBandStrategy("relative", threshold=0.01)` 或
`HedgeBandStrategy("sigma", k=0.5)`。旧的 `PriceIntervalStrategy` 与
`SigmaBandStrategy` 仍作为兼容入口保留。
在同一个参考价格和年化波动率下，三种单位可以互相换算：
`relative = absolute / S_ref`，
`sigma_multiple = absolute / (S_ref × sigma_annual / √ANNUAL_DAYS)`，因此也有
`sigma_multiple = relative / (sigma_annual / √ANNUAL_DAYS)`；当前项目统一使用
`ANNUAL_DAYS=243`。这只是参考点上的阈值对应关系，并不表示三种执行口径在整条
动态路径上严格等价：绝对口径保持固定价差，相对口径随上次实际对冲价计算，σ
口径还会按所选 implied / realized 来源和历史波动率窗口动态计算。
GUI 会同时显示绝对间隔、相对间隔和日波动 σ 倍数；编辑任意一项后按回车或
移出输入框，即会根据当前期权的 `s0` 与年化 `sigma` 自动反推另外两项；
修改参考参数也会自动刷新。普通单次回测始终按用户最后编辑的原始单位执行。
CSV/Wind 将期权从参考价伸缩到真实首价时，绝对带宽会按同一比例伸缩；相对值
和 σ 倍数保持不变。固定时刻模式要求真实日内 `DatetimeIndex`，且每个
纳入回测的交易日组都必须覆盖配置的全部时刻，因此适用于完整的 Wind 分钟行情
或带完整时间戳的日内 CSV。

GUI 只提供上述三种业务策略，默认使用每日收盘；多策略比较也只比较这三类
（行情不具备真实日内时刻时会跳过固定时刻）。旧的 `FixedFreqStrategy` 与
`hedge_freq` 参数仅作为后端 API 兼容入口保留，不再出现在 GUI 或默认策略比较中。
模拟来源可单独设置「模拟采样 bar/日」，用于控制路径离散化；CSV / Wind 不允许
手填每日 bar 数，底层会按真实时间索引和交易 session 自动推导。

`pricing.hedge_analysis` 提供 `compare_strategies` 批量分析策略，并提供
`summarize_strategy_result` 将已经完成的单次回测转换为统一指标；同时用
`recommend_by_rolling_history` 在多个相同期权期限的真实历史窗口上输出
近一周、月、季度、年的择优结果与完整排名。所有 bar 损益会先按真实交易日聚合，
默认分数为日净 PnL RMS（金额口径、已包含交易成本），越低越优。

GUI 的“回测结果对比”是独立的会话内结果池：每次普通回测成功后，用户可显式
命名并保留轻量结果快照；修改策略或参数继续回测不会覆盖旧快照。进入该页面后，
点击结果池“显示”列即可即时更新累计净 PnL、四张摘要卡、排名和相对最优差异，
整个过程只读取已完成结果，不重新运行回测。快照可重命名、删除；行情路径、
期权参数或头寸 / 成本口径不一致时仍允许并列查看，但界面会提示排名仅供场景
对照。该页面只管理普通回测快照，不承载历史搜索参数或历史排名。结果池仅保存
在当前应用会话，关闭程序后清空。

“◷ 历史择优”是与单次回测、回测结果对比平级的独立页面。候选策略开关、固定时刻
列表、固定间隔候选及 σ 来源 / 窗口均在该页面配置，不再混入
单次回测参数区。它复用统一的指标与回测引擎，在 CSV / Wind 真实历史上主动运行
近周、月、季度、年的滚动批量回测，并固定以日净 PnL RMS（已含成本）排序。
固定间隔候选统一用日波动 σ 倍数表达，默认
为 `0.5, 0.75, 1, 1.5, 2`，用户可编辑逗号列表；留空表示不额外加入常用档位。
启用“加入当前回测带宽”时，当前绝对 / 相对 / σ 输入会按当前 `s0` 与年化
`sigma` 换算成 σ 候选后加入。所有候选按 10 位有效数字规范去重，包含当前输入
在内最多 10 档。

历史择优中的固定间隔案例统一按 `sigma` 策略执行，并沿用界面选择的 σ 来源与
历史波动率窗口，使不同价格水平的滚动窗口共享同一波动尺度；这不会改变普通单次
回测按原始输入单位执行的行为。模拟来源会在入口直接禁用或拒绝，不启动任务，也
不会生成“历史推荐”。真实历史读取或滚动推荐计算失败时，整项任务失败，不会
退化为仅展示当前路径后仍提示完成。

历史排名与普通回测快照口径不同，因此不会直接混排。历史页提供“应用选中策略到
回测”和“当前路径验证并保留到对比”两个衔接动作：前者只回填策略参数，后者按
当前单次回测环境执行推荐策略，并把完成后的普通快照送入结果池。样本不足不是
计算失败：此时正式推荐可以为空，只要仍能返回非空诊断排名且至少有一个有效滚动
窗口，任务就会成功并标记“诊断领先”，同时显示有效 / 候选 / 跳过窗口与历史覆盖；
若一个有效窗口都无法形成，则会提示扩大真实历史区间。

## 📖 详细文档

完整的 GUI 操作手册（含每个参数的含义、单位、默认值与对结果的影响）请参阅：

👉 [**docs/GUI_USAGE.md**](docs/GUI_USAGE.md)

## 💬 反馈与贡献

发现 bug、想提需求或想贡献代码，欢迎开 [GitHub Issue](https://github.com/Grefer/DeltaLab/issues) 或直接提 PR。

## 📄 License

[MIT License](LICENSE) © Grefer
