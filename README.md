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
- **统一仓位方向**：左侧买入 / 卖出是 GUI 全部回测的唯一方向来源；单次、多路径、历史 C2C 与全部候选在任务启动时冻结同一方向，数量只能使用正数
- **可选收盘兜底**：左侧对冲策略区提供唯一公共开关，普通回测与历史择优共享每日收盘 Delta 对齐规则；原策略已在同一根收盘 bar 触发时自动去重
- **智能行情粒度**：Wind 支持“自动（推荐）”采样粒度，按策略选用日频或兼容的日内 bar；CSV / Wind 真实行情仍按时间索引与交易 session 自动推导每日 bar 数
- **回测结果池对比**：逐次回测后按需保留结果，买卖方向分组，以同方向最早保留的 close-to-close 快照为基准，再勾选其它快照即时比较
- **独立历史择优**：近周、近月、近季、近半年、近年五档周期可复选（默认全选、至少一项），使用 CSV / Wind 真实历史严格连续回放所选周期，并提供固定 C2C 加多个候选的完整 `L` 日累计 PnL 主视图
- **独立路径采样**：独立 `mc_seed` 多路径采样，避免样本相关性过高
- **完整可视化**：6 宫格对冲图表、波动率分析、蒙特卡洛盈亏分布、结构扫描；每日明细只展示并导出对冲触发记录
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
├── history_selection.py  # 策略优选纯逻辑 (候选空间 / 校验 / 排名与图表模型)
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

## ⚖️ 对冲模式与策略比较

### 三种触发方式

| GUI 选项 | 策略类 | 触发时点 | 行情要求 |
|---|---|---|---|
| 每日收盘 | `CloseToCloseStrategy` | 交易日组的最后一根真实 bar（非硬编码 `15:00`） | 任意 |
| 每日固定时刻 | `FixedTimeStrategy(["23:00","11:30","15:00"])` | 指定 `HH:MM`，按同一交易日「前一自然日夜盘 → 次日日盘」排列 | 真实日内 `DatetimeIndex` |
| 固定间隔 | `HedgeBandStrategy("absolute" / "relative" / "sigma", …)` | 相对上次实际对冲价越过带宽 | 任意 |

Wind 会按品种交易时段自动跳过明确落在休市区间的目标时刻（如无夜盘品种的 `23:00`），并在结果中披露请求、生效与跳过的时刻；仍在有效时段的目标必须在每个交易日组都有 bar，缺失即报错。CSV 没有可靠的交易时段元数据，因此保持严格模式，不会凭「历史中没出现过」猜测为休市。

> `PriceIntervalStrategy` / `SigmaBandStrategy` / `FixedFreqStrategy`（`hedge_freq`）仅作为后端 API 兼容入口保留，不出现在 GUI 与策略比较中。

### 带宽三种口径的换算

同一参考价 `S_ref` 与年化 `sigma` 下（项目统一 `ANNUAL_DAYS=243`）：

```text
relative       = absolute / S_ref
sigma_multiple = absolute / (S_ref × sigma_annual / √ANNUAL_DAYS)
               = relative / (sigma_annual / √ANNUAL_DAYS)
```

GUI 三个输入框联动：编辑任意一项后回车或移出输入框，即按当前 `s0` 与 `sigma` 反推另外两项。

> ⚠️ 换算只在参考点成立，**不代表三种口径在整条路径上等价**——绝对口径保持固定价差，相对口径相对上次实际对冲价计算，σ 口径还会按所选来源与回看期动态变化。单次回测始终按用户最后编辑的原始单位执行。CSV / Wind 把期权缩放到真实首价时，绝对带宽按同比例伸缩，相对间隔与 σ 倍数保持不变。

### 两个全局开关

- **每日收盘兜底对冲** — 左侧唯一的公共开关，普通回测与历史择优共用（择优启动时冻结当前值并应用到全部候选）。开启后每个**非到期、非敲出**交易日组的最后一根 bar 保证一次 Delta 对齐；原策略已在该 bar 触发时合并为一次，不重复调仓或重复计费。到期 / 敲出所在的末 bar 不做兜底，直接把标的头寸平至 `0`。
- **头寸方向** — `1` = 卖出期权，`-1` = 买入期权，是所有 GUI 回测的唯一方向来源，任务启动时冻结，历史 C2C 与全部候选共享。`quantity` 必须为有限正数，不能用负数量隐式翻转方向。结果内部保留模型原始 Greeks，展示 `-position × quantity × raw Greek` 的组合仓位 Greeks。

Wind 单次回测以**数据截止日**为日期主控：默认取最近一个已收盘交易日，再按期权剩余期限沿真实交易日历（`data/tradingday.csv`，离线优先）往前倒推建仓日，于是默认跑的就是截止日之前最近一个完整期限；取消勾选即可手工指定建仓日。行情采样粒度由策略唯一确定（收盘 = 日频、固定时刻 = 能覆盖目标时刻的最粗分钟、固定间隔 = 推荐日内粒度），界面只读披露、不提供手工调粗——手动调粗只会静默漏掉 bar 内触发、让结论偏乐观。

### 结果对比 vs 策略优选

| | 🆚 结果对比 | 🎯 策略优选 |
|---|---|---|
| 数据 | 会话内手动保留的回测快照 | CSV / Wind 真实历史的严格连续回放 |
| 基准 | 同方向最早保留的 `close_to_close` 快照 | 每日收盘 C2C，固定且不可关闭 |
| 排名 | 日净 PnL RMS | 增量收益（默认）/ 增量信噪比，点表头切换 |
| 是否重跑回测 | 否，只读已完成结果 | 是 |
| 生命周期 | 关闭程序即清空 | 可显式保存结果文件，重开后仍能载入 |

两者口径不同，不会混排。买入与卖出结果不进入同一排名；缺少同方向每日收盘快照时只展示绝对排名并明确提示。行情路径或成本口径不一致时仍允许并列查看，但界面会提示排名仅供场景对照。

### 策略优选的关键口径

- **周期** — 近周 / 近月 / 近季 / 近半年 / 近年可复选（默认全选、至少保留一项）。每档从共同分析截止日向前**连续**回放最近 `L` 个交易日，而不是在区间内挑若干终点再向前扩出完整期限。
- **代理段** — `L <= T` 时整段作为一个代理段、区间末按剩余期限公允价值 MTM；`L > T` 时按 `T` 续接，完整段到期结算、不足 `T` 的尾段 MTM。`T` 只定义每段最长存续期，不扩大证据区间。
- **排名** — 单一标的按增量金额：`增量收益 = Σ(候选日净 PnL − C2C 日净 PnL)`、`增量信噪比 = mean(逐日增量) / std(逐日增量)`，两者一并算出。品种合约池跨合约金额尺度不可相加，改按逐段有界 RMS 改善、以实际评分日加权，此时增量三列显示 `—`、表头不可点。
- **代码语义** — `P2609.DCE` 视为明确的单个合约，只分析它自身；`P.DCE` 视为品种入口，按 Wind `trade_hiscode` 逐日主力映射，只加载证据区间内真实出现过的历史合约，不引入截止日后才成为主力的合约。具体期货合约用不复权行情，股票 / ETF 用前复权。
- **候选带宽** — 统一用日波动 σ 倍数表达（默认 `0.5, 0.75, 1, 1.5, 2`），σ 恒取左侧输入的波动率，使不同价格水平的代理段共享同一波动尺度；含「加入当前回测带宽」在内最多 10 档，按 10 位有效数字去重。
- **资格** — 候选须覆盖严格区间内全部 C2C 代理段才具备正式择优资格，历史不足时只作诊断；候选均不优于基准（含持平）时明确显示「维持每日收盘」。
- **衔接动作** — `加载分段到展示页`（按重放配方重跑选中分段并渲染到摘要 / 图表 / 每日明细，可逐条查看建仓、触发与终止记录）、`仅回填参数`、`当前路径验证`（按排名表勾选批量执行，自动命名保留快照并切到结果对比页）。
- **图表** — 排名表首列最多勾选 8 个候选（默认勾选前三名里可比的候选），主图按严格区间顺序拼接全部非重叠共同代理段的日 PnL；共同交集不足 `L` 日时明确标注实际日数，**失败段不补零**。
- 模拟行情不支持策略优选；真实历史读取或严格区间计算失败时整项任务失败，不会退化为「仅展示当前路径」。

> 历史指标只用于参考，不构成置信区间、显著性结论或未来表现保证。

### Python API 入口

`pricing.hedge_analysis` 提供：

| 函数 | 用途 |
|---|---|
| `compare_strategies` | 批量比较多个策略 |
| `summarize_strategy_result` | 把已完成的单次回测转换为统一指标 |
| `recommend_by_rolling_history` | 单一标的在严格尾部历史区间上的择优结果与完整排名 |
| `recommend_by_contract_history_pool` | 期货品种池按历史具体合约边界完成同样的比较 |

逐字段说明见 [docs/GUI_USAGE.md](docs/GUI_USAGE.md)：§4.4 对冲参数、§6.7 回测结果对比、§6.8 历史择优。

## 📖 详细文档

完整的 GUI 操作手册（含每个参数的含义、单位、默认值与对结果的影响）请参阅：

👉 [**docs/GUI_USAGE.md**](docs/GUI_USAGE.md)

## 💬 反馈与贡献

发现 bug、想提需求或想贡献代码，欢迎开 [GitHub Issue](https://github.com/Grefer/DeltaLab/issues) 或直接提 PR。

## 📄 License

[MIT License](LICENSE) © Grefer
