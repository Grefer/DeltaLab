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
- **4 种对冲触发方式**：固定 bar 间隔 / 每日收盘 / 每日固定时刻 / 价格或日波动 σ 带宽
- **日内多频率**：支持 Daily / 60 min / 5 min / 1 min 等粒度；真实行情会按交易日组自动推导每日 bar 数，也可显式覆盖
- **策略对比与历史推荐**：同路径并列比较多种策略，并基于滚动历史给出近周、月、季度、年的推荐与样本诊断
- **独立路径采样**：独立 `mc_seed` 多路径采样，避免样本相关性过高
- **完整可视化**：6 宫格对冲图表、波动率分析、蒙特卡洛盈亏分布、结构扫描，支持每日明细导出
- **真实行情缩放**：CSV / Wind 模式下，以首日价为锚点等比例自动缩放期权参数（行权价、障碍价、赔付金额等），保持结构的相对一致性

## 📸 界面预览

![DeltaLab GUI 示意图](assets/screenshot.png)

> 左侧为参数面板，右侧为回测摘要 / 对冲图表 / 波动率分析 / 盈亏分布等标签页。

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
5. 查看 `回测摘要` / `对冲图表` / `波动率分析` / `盈亏分布` 等标签页

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
三种单位会统一换算为绝对价格带：`relative = absolute / S_last`，
`sigma_multiple = absolute / (S_last × sigma_annual / √ANNUAL_DAYS)`，因此也有
`sigma_multiple = relative / (sigma_annual / √ANNUAL_DAYS)`；当前项目统一使用
`ANNUAL_DAYS=243`。
GUI 会同时显示绝对间隔、相对间隔和日波动 σ 倍数；编辑任意一项后按回车或
移出输入框，即会根据当前期权的 `s0` 与年化 `sigma` 自动反推另外两项；
修改参考参数也会自动刷新。CSV/Wind 将期权从参考价伸缩到真实首价时，绝对
带宽也按同一比例伸缩，确保它与相对值、σ 倍数在单次回测、策略对比和每个
滚动历史窗口中保持同一口径。固定时刻模式要求真实日内 `DatetimeIndex`，且每个
纳入回测的交易日组都必须覆盖配置的全部时刻，因此适用于完整的 Wind 分钟行情
或带完整时间戳的日内 CSV。

`pricing.hedge_analysis` 提供 `compare_strategies` 批量运行和选择多个策略，
并用 `recommend_by_rolling_history` 在多个相同期权期限的真实历史窗口上输出
近一周、月、季度、年的推荐与完整排名。所有 bar 损益会先按真实交易日聚合，
默认分数为日净 PnL RMS（已包含交易成本），越低越优；样本不足时仅保留诊断
排名，不生成正式推荐。GUI 的“多策略对比 / 历史推荐”页支持多选策略查看差异。

## 📖 详细文档

完整的 GUI 操作手册（含每个参数的含义、单位、默认值与对结果的影响）请参阅：

👉 [**docs/GUI_USAGE.md**](docs/GUI_USAGE.md)

## 💬 反馈与贡献

发现 bug、想提需求或想贡献代码，欢迎开 [GitHub Issue](https://github.com/Grefer/DeltaLab/issues) 或直接提 PR。

## 📄 License

[MIT License](LICENSE) © Grefer
