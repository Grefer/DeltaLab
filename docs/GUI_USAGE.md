[← 返回项目 README](../README.md)

# DeltaLab GUI 使用文档

本文档面向首次使用 `gui_app.py` 的用户，目标是：能够从零启动 GUI、跑通一次完整回测、并理解界面上每一个参数的含义、单位、默认值与对结果的影响。

---

## 1. 环境与依赖

### 1.1 Python 版本

`pricing/Option_AS.py` 使用了 `match/case` 语法，需要 **Python 3.10+**。建议 3.11 或更高版本。

### 1.2 必装第三方库

| 库 | 用途 |
|---|---|
| `numpy` | 全部数值计算 |
| `scipy` | `Option_Vanilla` 用 `scipy.stats.norm`；GUI 波动率分布图用 `scipy.stats.norm` |
| `pandas` | 结果表格、Wind/CSV 数据读取 |
| `matplotlib` | 图表展示（GUI 强制使用 `TkAgg` 后端，见 `gui_app.py:19`） |
| `Pillow` (`PIL`) | 仅用于 `tools/make_icon.py` / `make_banner.py` / `make_workflow.py` 生成图标与 README 资源；运行 GUI 本身不需要 |
| `tkinter` | GUI 框架，标准库自带；个别 Linux 发行版需额外安装 `python3-tk` |

### 1.3 可选依赖

| 库 | 触发条件 | 不装时的影响 |
|---|---|---|
| `WindPy` | 在「数据来源」选 `Wind` | 选 Wind 模式运行时会抛 `ImportError`（提示见 `pricing/wind_data.py:125-129`），需安装 Wind 终端 + Python 插件，并已登录 |

CSV 与模拟模式无 Wind 依赖，可在普通环境下完整使用。

### 1.4 字体

GUI 在启动时会按平台自动挑选可用 CJK 字体（`gui_app.py:24-49`）：

- Windows：`Microsoft YaHei` / `SimHei` / `SimSun`
- macOS：`PingFang SC` 等
- Linux：`Noto Sans CJK SC` 等

若系统找不到任何 CJK 字体，matplotlib 会回退到 `DejaVu Sans`，中文会显示为方块；属于显示问题，不影响数值结果。

### 1.5 窗口图标与资源文件

GUI 启动时会通过 `BacktestApp._apply_window_icon()`（`gui_app.py:376-394`）自动加载 [assets/](../assets/) 下的图标：

- Windows 优先走 `assets/deltalab.ico`（16/32/48/64/128/256 多尺寸），在任务栏/标题栏表现最佳。
- macOS / Linux 或 Windows 上 `.ico` 加载失败时，回退到 `assets/deltalab.png` + `iconphoto()`。
- 两个文件都缺失时静默跳过，不影响主流程。

图标、banner、工作流示意图都是用 [tools/](../tools/) 下的脚本生成的（依赖 Pillow），可随时重跑调整外观：

```bash
python tools/make_icon.py      # 生成 assets/deltalab.png / .ico
python tools/make_banner.py    # 生成 assets/banner.png (README 顶部)
python tools/make_workflow.py  # 生成 assets/workflow.png (工作流示意图)
```

---

## 2. 启动方式

在仓库根目录执行：

```bash
python gui_app.py
```

入口为 `gui_app.py:2194` 的 `main()`。窗口默认大小 `1600x1000`，最小尺寸 `1200x720`（`gui_app.py:361-372`）。

### Windows 注意事项

- 路径中含中文一般没问题，但若遇到 Wind/CSV 加载失败，可优先排查路径。
- 高 DPI 屏幕下，左侧参数面板已包了 Canvas + Scrollbar（`gui_app.py:644-696`），可滚轮浏览，不会出现底部按钮被裁的情况。

---

## 3. 界面总览

主窗口分为左右两栏（`ttk.PanedWindow`，`gui_app.py:641`），可拖动中间分隔条。`_build_ui` 方法从 `gui_app.py:625` 开始。

### 3.1 左侧：参数与运行区

由上至下分四块：

1. **期权类型** —— 选大类与子类型（`gui_app.py:702`）
2. **期权参数** —— 随期权大类动态生成的输入框（`gui_app.py:725`）
3. **回测设置** —— 数据来源、对冲策略、滑点、合约乘数等（`gui_app.py:732`）
4. **运行按钮区** —— `运行回测` / `绘制结构图` / 进度条（`gui_app.py:930-955`）

### 3.2 右侧：结果标签页（`ttk.Notebook`，`gui_app.py:962`）

| Tab | 内容 |
|---|---|
| `回测摘要` | 单路径盈亏分解 + Greeks 统计 + 蒙特卡洛分布统计（若有） |
| `对冲图表` | 6 宫格：标的路径、Delta 与持仓、Gamma、Vega、Theta、累计盈亏 |
| `波动率分析` | 4 宫格：滚动 RV vs IV、累计 RV、价差、日收益分布 |
| `盈亏分布` | 仅模拟模式且 `n_paths>1` 时显示蒙特卡洛分布 |
| `结构分析` | 点 `绘制结构图` 后展示价格 / Greeks 对 S 的扫描曲线 |
| `每日明细` | DataFrame 表格 + `导出 CSV` 按钮 |

底部为状态栏（`gui_app.py:1053-1058`），实时显示「就绪 / 正在运行 / 完成」。

---

## 4. 参数详解

> 下文中所有参数名、控件值、字符串均直接对应 `gui_app.py` 当前代码，引用时用行内 code 格式。

### 4.1 期权类型

| 控件 | 选项 | 默认 | 说明 |
|---|---|---|---|
| `大类` | `香草期权 (Vanilla)` / `累计期权 (Decumulator)` / `亚式期权 (Asian)` / `气囊期权 (Airbag)` | `香草期权 (Vanilla)` | 见 `OPTION_CLASSES`，`gui_app.py:115` |
| `子类型` | 跟随大类切换 | 各大类的第一个 | 子类型驱动 `optiontype` 字段，决定走哪一种 payoff |

各大类的子类型集（`gui_app.py:116-213`）：

- **Vanilla**：`Eu`
- **Decumulator**：`Opt_Decumulator_Back` / `Opt_Decumulator_Fix` / `Opt_EnDecumulator` / `Opt_EnDecumulator_Fix` / `Opt_ASGQ_call_put` / `Opt_ASGQ_EP` / `Opt_ASGQ_EF` / `Opt_ASGQ_DP` / `Opt_ASGQ_DF`
- **Asian**：`Asian` / `EnhanceAsian`
- **Airbag**：`Opt_Airbag`

每个子类型的 payoff 公式见 `STRUCTURE_DOCS`（`gui_app.py:258` 起），也会在「结构分析」Tab 顶部展示。

### 4.2 期权参数（按大类分组）

每个字段都是一个 Entry，初始值取自 `OPTION_CLASSES[...]["params"]` 的 default。空字符串会被解释为 0（见 `_collect_gui_state`，`gui_app.py:1161`）。

#### 香草期权 (Vanilla)

| 参数 | 含义 | 类型 | 默认 |
|---|---|---|---|
| `s0` | 初始价格 S0 | float | `100.0` |
| `K` | 行权价 | float | `100.0` |
| `T_days` | 期限（交易日） | int | `22` |
| `sigma` | 波动率（年化，小数） | float | `0.18` |
| `cp` | 方向，`1`=Call，`-1`=Put | int | `1` |
| `r` | 无风险利率 | float | `0.03` |
| `q` | 分红率 | float | `0.03` |

定价走 BS 封闭解（`Option_Vanilla.blsprice`）。

#### 累计期权 (Decumulator)

| 参数 | 含义 | 类型 | 默认 |
|---|---|---|---|
| `s0` | 初始价格 | float | `100.0` |
| `K` | 行权价 | float | `90.0` |
| `T_days` | 剩余期限（交易日） | int | `20` |
| `T_over` | 已过天数 | int | `0` |
| `sigma` | 波动率 | float | `0.18` |
| `H` | 障碍价格（敲出/熔断） | float | `110.0` |
| `N` | 杠杆倍数（跌破 K 后） | int | `2` |
| `cp` | 方向 | int | `1` |
| `fix` | 固定赔付（`*_Fix` 子类型用） | float | `0.0`，传 0 → 视为未设置（None） |
| `P` | 保障价格（`ASGQ_*P` 用） | float | `0.0`，传 0 → None |
| `amount` | 固定金额（`ASGQ_*F` 用） | float | `0.0`，传 0 → None |
| `r` / `q` | 利率 / 分红 | float | `0.03` / `0.03` |
| `nPath` | MC 路径数 | int | `100000` |

注意 `fix / P / amount` 的「0 当 None」是在 `OPTION_CLASSES["累计期权"]["build"]` 里做的（`gui_app.py:163-165`），对应不需要这些字段的子类型可以保留 0。

#### 亚式期权 (Asian)

| 参数 | 含义 | 类型 | 默认 |
|---|---|---|---|
| `s0` | 初始价格 | float | `100.0` |
| `K` | 行权价 | float | `100.0` |
| `E` | 增强价（仅 `EnhanceAsian` 用） | float | `100.0` |
| `T` | 期限（交易日） | int | `22` |
| `N` | 观察日数（取末 N 日均价） | int | `22` |
| `sigma` | 波动率 | float | `0.15` |
| `cp` | 方向 | int | `1` |
| `minPay` | 最低赔付 | float | `0.0` |
| `maxPay` | 最高赔付 | float | `999999.0`（实际充当 `+∞`） |
| `r` / `q` | 利率 / 分红 | float | `0.03` / `0.03` |
| `nPath` | MC 路径数 | int | `100000` |

#### 气囊期权 (Airbag)

| 参数 | 含义 | 类型 | 默认 |
|---|---|---|---|
| `s0` | 初始价格 | float | `100.0` |
| `K` | 行权价 | float | `100.0` |
| `KI` | 敲入价 | float | `90.0` |
| `T_days` | 期限（交易日） | int | `20` |
| `sigma` | 波动率 | float | `0.18` |
| `pr` | 未敲入参与率 | float | `0.8` |
| `pr_ki` | 已敲入参与率 | float | `1.0` |
| `cp` | 方向 | int | `1` |
| `r` / `q` | 利率 / 分红 | float | `0.03` / `0.03` |
| `nPath` | MC 路径数 | int | `100000` |

观察日默认取 `range(1, T_days+1)`（每日观察），见 `OPTION_CLASSES["气囊期权"]["build"]`（`gui_app.py:207`）。

### 4.3 回测设置 — 数据来源

`数据来源` 是单选 Radiobutton（`gui_app.py:740-745`），三选一：

| 值 | 标签 | 默认 |
|---|---|---|
| `simulate` | 模拟 | ✓ |
| `csv` | CSV |  |
| `wind` | Wind |  |

切换数据来源时，`_toggle_source`（`gui_app.py:1087`）会动态显示对应参数子区，并对 csv / wind 强制锁定 `steps_per_day=1`（实盘/CSV 仅支持日频）。

#### 模拟（simulate）

| 字段 | 含义 | 默认 |
|---|---|---|
| `种子` | `numpy` `default_rng` 的基础种子 | `42` |
| `已实现波动率` | 用于生成路径的真实 σ；空 = 用期权 `sigma`（隐含） | `""`（空） |
| `模拟路径数` | MC 路径数。`>1` 时启用蒙特卡洛多路径分析；其中第一条用作主单路径展示 | `10` |

> 调高 `模拟路径数` 会显著放慢运行（路径数 × 单路径回测耗时），但能稳定盈亏分布、对冲误差等统计量。
>
> **Per-path MC 种子**：多路径模式下每条路径都有独立的 `option.mc_seed = base_seed + path_idx`（`hedge_backtest.py:277-282`），避免所有路径共用同一批 MC 采样而人为压窄分布。整体仍可复现（固定 `base_seed` 即可），但子样本之间相互独立。

#### CSV

| 字段 | 含义 | 默认 |
|---|---|---|
| `文件` | CSV 路径，旁边有 `浏览…` 按钮 | 空 |
| `价格列` | CSV 中收盘价列名 | `close` |

底层走 `HedgeBacktest.from_csv`（`hedge_backtest.py:1156`）。日期列默认取第一列（`pd.read_csv(parse_dates=[0])`）。

#### Wind

| 字段 | 含义 | 默认 |
|---|---|---|
| `代码` | Wind 标的代码 | `510050.SH` |
| `起始日` | 建仓日（含） | 启动当日往前推 90 天，格式 `YYYY-MM-DD` |
| `结束日` | 结束日（含） | 启动当日，格式 `YYYY-MM-DD` |

底层走 `HedgeBacktest.from_wind`（`hedge_backtest.py:977`），复权方式硬编码为 `"F"`（前复权）。

> 真实行情模式下，期权参数中的 `s0` 视为「参考价 S_ref」；GUI/底层会按 `ratio = 真实首日价 / S_ref` 自动缩放 `K / KI / H / P / fix / amount` 等价格量纲字段，保证期权结构不被破坏。详细缩放表会展示在「回测摘要」Tab 中（由 `_show_summary` 渲染），逻辑在 `hedge_backtest.py:_rescale_option_to_real_s0`（`hedge_backtest.py:181`）。

### 4.4 回测设置 — 对冲参数

| 字段 | 含义 | 单位 / 取值 | 默认 |
|---|---|---|---|
| `调仓频率(天)` | `FixedFreqStrategy` 的 bar 间隔；`sigma_band` 模式下被忽略（但仍会读取），回测摘要中也不再显示 | int ≥ 1 | `1` |
| `交易成本率(%)` | 单边费率，按成交额收取；GUI 输入百分比，内部除以 100 | float，单位 % | `0.01`（即 0.0001） |
| `头寸方向` | `1`=卖出（short），`-1`=买入（long） | Radiobutton | `1`（卖出） |
| `交易数量` | `quantity`，将 Δ 转换为标的份数 | float | `100` |
| `合约乘数` | `multiplier`，每手对应的标的数量；`0` = 不取整 | float ≥ 0 | `5` |

> GUI 默认值 `5` 与底层 `HedgeBacktest.__init__(multiplier=5)` 一致，语义是「每 5 个标的为一手，按手数取整」。如果你想完全连续对冲，请填 `0`。

#### 4.4.1 对冲策略（高级）

| 字段 | 含义 | 选项 | 默认 |
|---|---|---|---|
| `对冲策略` | 调仓触发方式 | `fixed_freq` / `sigma_band` | `fixed_freq` |

切到 `sigma_band` 时，下方多出 3 个字段（`gui_app.py:884-902`），切换由 `_toggle_strategy`（`gui_app.py:1130`）控制：

| 字段 | 含义 | 取值 | 默认 |
|---|---|---|---|
| `调仓带宽 k·σ` | 触发阈值倍数 | float > 0 | `0.5` |
| `σ 来源` | 用什么 σ 估计触发阈值 | `implied` / `realized` | `implied` |
| `历史波动率窗口 N (日)` | `realized` 时滚动窗口长度（单位=日） | int ≥ 2 | `20` |

触发条件（来自 `SigmaBandStrategy.should_hedge`，`hedge_backtest.py:140` 起）：

```
|ln(S / S_last)| >= k * σ_ref * sqrt(dt_since_last)
```

其中 `σ_ref` 取 `option.sigma`（implied）或最近 `window_days` 个交易日对数收益的年化 std；`realized` 模式下样本不足时回退到 implied。

`fixed_freq` 模式下，`SigmaBandStrategy` 控件被隐藏，调仓频率取自 `调仓频率(天)`（每 N 个 bar 调一次仓，`FixedFreqStrategy`，类定义于 `hedge_backtest.py:48`）。

#### 4.4.2 每日 bar 数（intraday）

| 字段 | 含义 | 选项 | 默认 |
|---|---|---|---|
| `每日 bar 数` | `steps_per_day`，把 1 个交易日切成 N 根 bar | `1` / `4` / `48` / `240` | `1` |

下拉提示文字：`1=日频 / 4=60分 / 48=5分 / 240=1分`（`gui_app.py:914`）。

- `1`（默认） → 行为与传统日频一致。
- `4` → 每日 4 根 60 分 bar。
- `48` → 每日 48 根 5 分 bar。
- `240` → 每日 240 根 1 分 bar。

> CSV / Wind 模式下该控件被禁用并强制为 `1`（实盘/CSV 仅支持日频，`gui_app.py:_toggle_source`、`hedge_backtest.py:1055`（`from_wind`）/ `hedge_backtest.py:1197`（`from_csv`））。

跨日 bar 上 `option.step_forward` 被调用；日内 bar 仅用 `_bumped_copy(_intraday_elapsed=…)` 临时评估 Δ，不污染 option 内部状态（`hedge_backtest.py:531-532`）。年化口径 `dt_bar = 1 / (ANNUAL_DAYS * spd)`，已实现波动率年化因子 `ANNUAL_DAYS * spd`。

#### 4.4.3 滑点

| 字段 | 含义 | 单位 | 默认 |
|---|---|---|---|
| `滑点 (bps)` | `slippage_bps`，单边滑点（万分之一为 1 bps） | bps | `0` |

实现（`hedge_backtest.py:515-596`）：

- 买入（`trade > 0`）成交价上浮 `bps × 1e-4`；卖出下浮。
- TC 同时计入 `|trade| × s_exec × tc_rate` 与 `|trade| × S × sl_rate` 两部分（手续费和滑点不重复计）。
- Day 0 建仓也按这个口径收滑点（`hedge_backtest.py:494-507`）。

### 4.5 运行按钮区

| 按钮 | 行为 |
|---|---|
| `▶  运行回测` | 主流程：构建期权 → 生成/读取价格 → 执行 `HedgeBacktest.run()`，模拟模式 `n_paths>1` 时还会跑 `run_multi` 并显示进度条 |
| `📊  绘制结构图` | 不依赖回测结果，扫描 S 在 `[s0×(1−r), s0×(1+r)]` 区间，展示 Price / Δ / Γ / ν / Θ 曲线 |
| `扫描 ±%` | 结构图扫描幅度，默认 `30`（即 ±30%），合法范围 `(0, 100)` |
| `点数` | 扫描点数，默认 `31`，合法范围 `5~201` |

> 结构图默认会限制 `nPath ≤ 20000` 以加速扫描（见 `_structure_worker`，`gui_app.py:1983`），原值若大于 20000 仅在结构图时生效，回测主流程仍用原值。
>
> 结构图视角与 `头寸方向` 联动：卖方（`position=1`）时 `sign=-1`，所有曲线乘以 `sign`（见 `_show_structure`，`gui_app.py:2041`），方便直观判断卖方账户的盈亏方向。

---

## 5. 一次完整回测的操作流程

以「卖出 22 日香草 ATM Call、模拟 500 条路径」为例：

1. 打开 `期权类型` → `大类` 选 `香草期权 (Vanilla)`，`子类型` 自动 `Eu`。
2. `期权参数` 区保留默认（s0=100, K=100, T_days=22, sigma=0.18, cp=1, r=0.03, q=0.03）。
3. `回测设置`：
   - `数据来源` 选 `模拟`
   - `种子` `42`
   - `已实现波动率` 留空（=隐含）
   - `模拟路径数` 改为 `500`
   - `调仓频率(天)` `1`，`交易成本率(%)` `0.01`
   - `头寸方向` `卖出 (short)`
   - `交易数量` `100`，`合约乘数` `0`（连续对冲）
   - `对冲策略` `fixed_freq`，`每日 bar 数` `1`，`滑点 (bps)` `0`
4. 点 `▶  运行回测`。底部状态栏显示「正在运行回测…」，进度条先显示不定模式，进入 MC 阶段后切换为确定模式（带 `蒙特卡洛模拟: x/500` 文字）。
5. 完成后状态栏变为「回测完成 | 切换上方 Tab 查看各项结果」，自动跳到「回测摘要」Tab。

预期看到：

- **回测摘要**：含期权初始/到期价值、对冲盈亏、MtM 盈亏、对冲误差、Greeks 表格；底部追加蒙特卡洛分布（均值/分位数/盈利概率/RV 统计）。
- **对冲图表**：6 宫格曲线。
- **波动率分析**：4 宫格，含 RV vs IV、价差填色、日收益直方图叠正态。
- **盈亏分布**：直方图 + 散点，含线性拟合斜率（盈亏 vs 波动率价差）。
- **每日明细**：含 13 列的 Treeview 表格 + 导出按钮。

---

## 6. 结果解读

### 6.1 回测摘要（`gui_app.py:1423`，`_show_summary`）

- **单路径盈亏分解**：
  - `标的对冲盈亏` = `Σ H[i-1]·(S[i]-S[i-1])` —— 标的腿浮动盈亏
  - `期权 MtM 盈亏` = `Σ -position·(V[i]-V[i-1])·quantity` —— 期权方价值变动（卖方视角下取负）
  - `累计交易成本` = `Σ TC[i]`
  - `对冲误差` = `position·V[0]·quantity + Σ hedge_daily − Σ TC − position·V[-1]·quantity`，对应 `hedge_backtest.py:620`
- **波动率分析**：`成交隐含波动率` 直接来自 `option.sigma`；`已实现波动率` 用全样本 bar 级对数收益年化。
- **Greeks 统计**：表格列出 Delta/Gamma/Vega/Theta/Rho 的初始值、序列均值、最大绝对值。

### 6.2 对冲图表（`gui_app.py:1591`，`_show_chart`）

6 宫格，从左上到右下：标的价格、Delta 持仓目标 vs 实际持仓、Gamma、Vega、Theta、累计盈亏（绿/红填色表示正负）。

> Delta 持仓目标 = `delta × quantity × position`，实际持仓 `shares` 受 `multiplier` 取整影响，会出现明显阶梯。

### 6.3 波动率分析（`gui_app.py:1684`，`_show_vol_chart`）

- (1) 滚动 RV(20d) vs IV，填色表示 RV 与 IV 大小关系（红=RV>IV）。
- (2) 累计 RV 与 IV 的收敛对比。
- (3) 价差时间序列 = IV − 滚动 RV，正值（绿色）表示卖方相对有利。
- (4) 日收益直方图叠 IV 正态、RV 正态；其中日 IV σ = `IV / sqrt(ANNUAL_DAYS)` 转换为百分比（`ANNUAL_DAYS=243`，见 `pricing/constants.py`）。

### 6.4 盈亏分布（`gui_app.py:1784`，`_show_dist_chart`）

仅 `simulate + n_paths>1` 时有内容，否则显示提示文字。

- (1) 总盈亏直方图 + 均值线 + 5% VaR 线
- (2) 对冲误差直方图
- (3) 已实现波动率分布
- (4) 总盈亏 vs (IV−RV) 散点 + 线性拟合（斜率正常应为正：RV 比 IV 低越多，卖方赚得越多）

### 6.5 结构分析（`gui_app.py:2041`，`_show_structure`）

顶部有 `STRUCTURE_DOCS` 中的中文 payoff 描述 + 当前参数摘要；下方 6 宫格扫描曲线，含 `K/H/KI/E/P` 等关键价位的虚线标记（颜色区分）。视角受头寸方向控制：卖方（`position=1`）时 `sign=-1`，买方（`position=-1`）时 `sign=1`，所有曲线均乘以 `sign`，方便直观判断。

### 6.6 每日明细（`gui_app.py:1885`，`_show_table`）

Treeview 形式的每日表格，含 13 列：标的价格 / 期权价值 / Δ/Γ/ν/Θ/ρ / 持仓 / 标的盈亏 / 期权盈亏 / 交易成本 / 每日净盈亏 / 累计盈亏。可点 `导出 CSV` 用 `utf-8-sig` 编码保存（Excel 友好）。

---

## 7. 常见问题与坑

### 7.1 启动相关

- **`ImportError: WindPy`**：仅在选 Wind 数据源时触发（`pricing/wind_data.py:125-129`）。模拟和 CSV 模式不需要 Wind。
- **`ModuleNotFoundError: tkinter`**：部分 Linux 发行版默认不带 tkinter，安装 `python3-tk`（Debian/Ubuntu）或 `python3-tkinter`（RHEL）即可。Windows / macOS 官方 Python 自带。
- **`SyntaxError: invalid syntax (match/case)`**：Python 版本低于 3.10。

### 7.2 价格序列长度不足

报错示例：`价格序列长度不足：期权剩余 22 日 x steps_per_day=1 需要 23 个价格点，实际仅 18 个。`

来源：`hedge_backtest.py:424-425`。CSV / Wind 模式下需要保证起止日范围内的交易日数 ≥ 期权剩余期限。可以放宽日期区间或调小 `T_days / T`。

### 7.3 历史价 S_ref 必须 > 0

CSV / Wind 模式下若期权 `s0` 设为 0 或负数，会触发 `_rescale_option_to_real_s0` 抛错（`hedge_backtest.py:201`）。请填入正的参考价。

### 7.4 `每日 bar 数` 在 CSV / Wind 模式下被锁死

Wind / CSV 真实行情仅支持日频；`_toggle_source` 会强制 `spd_var=1` 并禁用下拉框，UI 提示文字变为「实盘/CSV 模式仅支持日频 (spd=1)」。即便用户绕过 GUI 直接调用 `from_wind(steps_per_day=4)`，底层也会强制改为 1 并打印日志（`hedge_backtest.py:1055`）。

### 7.5 蒙特卡洛盈亏分布 Tab 一片空白

只有 `数据来源 = 模拟` 且 `模拟路径数 > 1` 时才会触发 `run_multi`，否则该 Tab 只显示提示文字。

### 7.6 结构图扫描 nPath 被截断

`_structure_worker`（`gui_app.py:1983`）会把 `nPath > 20000` 强制截断为 20000 以加速扫描。这只影响结构图，不影响主回测中真实的 `nPath`。

### 7.7 `历史波动率窗口 N (日)` 样本不足

`SigmaBandStrategy(sigma_source='realized')` 在窗口内有效样本 < 2 时（例如刚开始几个 bar）会自动回退到 implied σ（`hedge_backtest.py:SigmaBandStrategy.should_hedge`），不会抛错。

### 7.8 `multiplier` 默认值 = 5

GUI 与底层 `HedgeBacktest.__init__` 均默认 `合约乘数 = 5`（`hedge_backtest.py:341`）。新手第一次跑可能看到「Delta 与实际持仓」图上有明显阶梯——这是按 5 取整造成的，并非 bug。如需连续对冲请改为 `0`。

---

## 相关源码

- [`gui_app.py`](../gui_app.py) — GUI 入口与全部控件
- [`pricing/hedge_backtest.py`](../pricing/hedge_backtest.py) — `HedgeBacktest` / `HedgeStrategy` / `FixedFreqStrategy` / `SigmaBandStrategy`
- [`pricing/option_base.py`](../pricing/option_base.py) — `OptionBase`，统一定价/Greeks 接口
- [`pricing/Option_Vanilla.py`](../pricing/Option_Vanilla.py) — 香草期权 + `blsprice`
- [`pricing/Option_AS.py`](../pricing/Option_AS.py) — 亚式期权
- [`pricing/Option_AB.py`](../pricing/Option_AB.py) — 气囊期权
- [`pricing/Option_DE.py`](../pricing/Option_DE.py) — 累计/熔断累计系列
- [`pricing/wind_data.py`](../pricing/wind_data.py) — Wind 数据接口（可选）
- [`pricing/mc_engine.py`](../pricing/mc_engine.py) — GBM 路径生成引擎（`McGbmQ`，quasi-MC 反对称采样）
- [`pricing/rolling_backtest.py`](../pricing/rolling_backtest.py) — 滚动回测（Phase 3）：按窗口滑动跑多期回测并汇总盈亏分布
- [`pricing/constants.py`](../pricing/constants.py) — `ANNUAL_DAYS = 243.0`
