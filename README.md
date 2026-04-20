# DeltaLab

> DeltaLab — 基于 Python / tkinter 的期权 Delta 动态对冲回测框架，支持多种奇异期权、多数据源接入与蒙特卡洛分析。

---

## ✨ 功能特性

- **4 大类期权**：香草 (Vanilla)、累计 (Decumulator)、亚式 (Asian)、气囊 (Airbag)，共 13 种子类型
- **3 种数据源**：蒙特卡洛模拟 / CSV 历史行情 / Wind 实时终端
- **2 种对冲策略**：固定频率 (`fixed_freq`) / σ-带触发 (`sigma_band`)
- **日内多频率**：支持日频 / 60 分 / 5 分 / 1 分级别的对冲模拟
- **完整可视化**：6 宫格对冲图表、波动率分析、蒙特卡洛盈亏分布、结构扫描、每日明细导出
- **真实行情缩放**：CSV / Wind 模式下自动按 `S_ref → S_real` 比例缩放期权要素，保持结构一致性

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

窗口默认大小 1600×1000，最小 1200×720。

### 回测步骤

1. **期权类型** → 选 `香草期权 (Vanilla)`
2. **期权参数** → 保留默认（ATM Call, 22 天, σ=0.18）
3. **回测设置** → 数据来源选 `模拟`，模拟路径数改为 `500`
4. 点击 **▶ 运行回测**
5. 查看 `回测摘要` / `对冲图表` / `波动率分析` / `盈亏分布` 等标签页

## 📁 项目结构

```
DeltaLab/
├── gui_app.py                  # GUI 入口（tkinter + matplotlib）
├── pricing/                    # 核心定价与回测引擎
│   ├── __init__.py
│   ├── constants.py            # ANNUAL_DAYS = 243.0
│   ├── option_base.py          # OptionBase 基类（有限差分 Greeks）
│   ├── Option_Vanilla.py       # 香草期权（Black-Scholes 封闭解）
│   ├── Option_AS.py            # 亚式期权（蒙特卡洛）
│   ├── Option_AB.py            # 气囊期权（蒙特卡洛）
│   ├── Option_DE.py            # 累计/熔断累计系列（蒙特卡洛）
│   ├── mc_engine.py            # GBM 路径生成引擎
│   ├── hedge_backtest.py       # HedgeBacktest + 对冲策略
│   ├── rolling_backtest.py     # 滚动回测（多窗口）
│   └── wind_data.py            # Wind 数据接口（可选）
├── tests/                      # 测试
├── data/cache/                 # Wind 数据缓存（运行时生成）
├── docs/
│   └── GUI_USAGE.md            # 完整 GUI 操作手册
└── archive/                    # 归档的旧版文件
```

## 📊 支持的期权类型
可以通过点击 **📊  绘制结构图** 查看期权结构说明
| 大类 | 子类型 | 定价方式 |
|---|---|---|
| 香草期权 (Vanilla) | `Eu` | Black-Scholes 封闭解 |
| 累计期权 (Decumulator) | `Opt_Decumulator_Back`, `Opt_Decumulator_Fix`, `Opt_EnDecumulator`, `Opt_EnDecumulator_Fix`, `Opt_ASGQ_call_put`, `Opt_ASGQ_EP`, `Opt_ASGQ_EF`, `Opt_ASGQ_DP`, `Opt_ASGQ_DF` | 蒙特卡洛 |
| 亚式期权 (Asian) | `Asian`, `EnhanceAsian` | 蒙特卡洛 |
| 气囊期权 (Airbag) | `Opt_Airbag` | 蒙特卡洛 |

## 🔧 技术栈

| 组件 | 技术 |
|---|---|
| GUI 框架 | tkinter + ttk（自定义现代扁平主题） |
| 图表 | matplotlib（TkAgg 后端） |
| 定价引擎 | numpy / scipy（BS 封闭解 + 蒙特卡洛） |
| 数据处理 | pandas |
| 实时行情 | WindPy（可选） |
| 并发 | threading + ThreadPoolExecutor |

## 📖 详细文档

完整的 GUI 操作手册（含每个参数的含义、单位、默认值与对结果的影响）请参阅：

👉 [**docs/GUI_USAGE.md**](docs/GUI_USAGE.md)

## 📄 License

[MIT License](LICENSE) © Grefer
