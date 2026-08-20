# _*_ coding: utf-8 _*_
"""
滚动历史回测驱动器（Phase 3）

给定一个期权配置和一条 Wind 历史行情，按固定步长向前滚动：
在每个窗口上用事前 HV 构造期权实例，分别跑
    1) 真实历史路径（path_source="historical"）
    2) 同 σ 生成的一条 GBM 对照路径（path_source="gbm"）
并将两边的对冲 PnL、离散化误差等汇总到一张 DataFrame。

语义说明（与 from_wind / from_csv 对齐）：
- 历史价格保留真实水平（不再 rebase 到 option_cfg['s0']）。
- option_cfg['s0'] 视为参考价 S_ref，仅用于配置 strike / barrier / cashflow
  等价格量纲字段。每一轮以该窗口的真实起始价 S_real 为基准，按
  ratio = S_real / S_ref 通过 ``_rescale_option_to_real_s0`` 缩放期权要素。
- 由于每轮 S_real 不同，每轮独立缩放（ratio 随窗口变化）。

依赖：
- pricing.wind_data.get_log_returns / get_contract_spec
- pricing.hedge_backtest.HedgeBacktest / _rescale_option_to_real_s0

注意：本模块不做期权类型特判，调用方负责传入正确的 option_cfg。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .constants import ANNUAL_DAYS
    from .hedge_backtest import HedgeBacktest, _rescale_option_to_real_s0
    from .wind_data import get_log_returns, get_contract_spec
except ImportError:  # 兼容脚本直接运行
    from constants import ANNUAL_DAYS
    from hedge_backtest import HedgeBacktest, _rescale_option_to_real_s0
    from wind_data import get_log_returns, get_contract_spec


def run_rolling_backtest(
    option_cfg: dict,
    option_class,
    code: str,
    start: str,
    end: str,
    step: int = 5,
    hv_window: int = 20,
    asset_type: str = "equity",
    r: float = 0.02,
    q: float = 0.0,
    hedge_kwargs: dict | None = None,
    mc_seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    按固定步长在历史行情上滚动回测期权的动态对冲表现。

    Parameters
    ----------
    option_cfg : dict
        期权构造参数字典，必须包含 ``s0`` 和 ``T``（存续期，交易日数）。
        其他字段原样透传给 ``option_class``；``sigma`` 会被每个窗口的事前 HV 覆盖。
    option_class : type
        期权类，例如 ``Option_Vanilla``。
    code : str
        Wind 标的代码。
    start, end : str
        历史数据起止日期 "YYYY-MM-DD"。
    step : int
        滚动步长（交易日），默认 5。
    hv_window : int
        事前 HV 窗口（交易日），默认 20。
    asset_type : str
        "equity" 或 "future"；"future" 会自动查询合约乘数并按 Black-76 处理（q=r）。
    r, q : float
        默认无风险利率与分红率。若 option_cfg 中已指定则以 option_cfg 为准。
        future 场景下会强制 q = r。
    hedge_kwargs : dict
        额外传给 ``HedgeBacktest`` 的关键字参数（如 ``hedge_freq``、``tc_rate``、
        ``position``、``quantity``、``multiplier``）。
    mc_seed : int
        MC 对照路径的基础种子，每个起点用 ``mc_seed + t0`` 派生。
    verbose : bool
        True 时打印每个跳过窗口的原因，以及最终跳过数量。

    Returns
    -------
    pd.DataFrame
        每行对应一个滚动起点，列包括 start_date / sigma_pre /
        hedge_pnl_real / hedge_pnl_mc / final_price_real / final_price_mc /
        delta_disc_real / delta_disc_mc。
    """
    # ---- 参数基本检查 ----
    if "s0" not in option_cfg or "T" not in option_cfg:
        raise ValueError("option_cfg 必须包含 's0' 与 'T'")

    T = int(option_cfg["T"])
    if T <= 0:
        raise ValueError(f"option_cfg['T'] 必须为正整数交易日，当前 {T}")

    # ---- 拉取历史对数收益 ----
    log_ret = get_log_returns(code, start, end, asset_type=asset_type)
    if not isinstance(log_ret, pd.Series):
        log_ret = pd.Series(log_ret)

    if hv_window + T > len(log_ret):
        raise ValueError(
            f"历史数据长度不足：hv_window({hv_window}) + T({T}) = "
            f"{hv_window + T} > len(log_ret)={len(log_ret)}"
        )

    # 同时拿到真实收盘价序列：保留真实价格水平。
    # log_ret 长度比 prices 少 1（dropna 第 0 行），对齐方式：
    #     log_ret.index[k] 对应 prices.index[k+1]
    # 在 Wind 缓存可用时优先取真实价格；若不可用（例如单测对 get_log_returns
    # 做了 patch 而真实数据不存在），则退化为以 cfg s0 为起点根据 log_ret
    # 重建价格（此时 ratio≈1，等价于旧的 rebase_path 行为）。
    cfg_s0_for_recon = float(option_cfg["s0"])
    prices_arr = None
    prices_index = None
    try:
        try:
            from .wind_data import load_history_cached
        except ImportError:
            from wind_data import load_history_cached
        prices_series = load_history_cached(code, start, end, asset_type)
        candidate = np.asarray(prices_series.values, dtype=float)
        if len(candidate) >= len(log_ret) + 1:
            prices_arr = candidate[-(len(log_ret) + 1):]
            prices_index = prices_series.index[-(len(log_ret) + 1):]
    except Exception:
        prices_arr = None
        prices_index = None

    if prices_arr is None:
        # 退化路径：用 log_ret 重建价格，起点 = 配置 s0
        cum = np.concatenate([[0.0], np.cumsum(log_ret.values)])
        prices_arr = cfg_s0_for_recon * np.exp(cum)

    if prices_index is None:
        # 与 prices_arr 逐格对齐的日期轴。退化路径拿不到首个价格的真实日期
        # （它落在 log_ret 首日的前一个交易日），补一个 NaT 占位即可——下面
        # 只会用 t0 >= 1 的格子。
        prices_index = log_ret.index.insert(0, pd.NaT)

    # ---- 期货 vs 权益：合约乘数与有效 q ----
    if asset_type == "future":
        spec = get_contract_spec(code)
        is_future = True
        contract_multiplier = float(spec.get("multiplier", 1.0))
        effective_q = float(option_cfg.get("q", r))  # Black-76：q = r
        if "q" not in option_cfg:
            effective_q = r
    else:
        is_future = False
        contract_multiplier = 1.0
        effective_q = float(option_cfg.get("q", q))

    effective_r = float(option_cfg.get("r", r))

    hk_base = dict(hedge_kwargs or {})
    hk_base.update(
        dict(
            is_future=is_future,
            contract_multiplier=contract_multiplier,
        )
    )
    # 方向只能由 position 表达。滚动驱动器会在循环内捕获单窗异常，
    # 因此必须在进入循环前校验，避免非法方向被静默吞掉后返回空结果。
    position = hk_base.get("position", 1)
    if isinstance(position, (bool, np.bool_)):
        raise ValueError("hedge_kwargs['position'] 只允许 1（卖出）或 -1（买入）")
    try:
        position_value = float(position)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "hedge_kwargs['position'] 只允许 1（卖出）或 -1（买入）"
        ) from exc
    if (not np.isfinite(position_value)
            or position_value not in (-1.0, 1.0)):
        raise ValueError(
            "hedge_kwargs['position'] 只允许 1（卖出）或 -1（买入）")
    position = int(position_value)
    hk_base["position"] = position

    quantity = hk_base.get("quantity", 1.0)
    try:
        quantity = float(quantity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("hedge_kwargs['quantity'] 必须为有限正数") from exc
    if not np.isfinite(quantity) or quantity <= 0:
        raise ValueError("hedge_kwargs['quantity'] 必须为有限正数")
    hk_base["quantity"] = quantity

    cfg_s0 = float(option_cfg["s0"])  # S_ref，仅用于按比例缩放期权要素
    dt = 1.0 / ANNUAL_DAYS

    rows = []
    skipped = 0
    first_rescale_info = None
    n_lr = len(log_ret)

    # 遍历所有可行滚动起点
    for t0 in range(hv_window, n_lr - T + 1, step):
        try:
            # 4.1 事前 HV（年化）
            pre_slice = log_ret.iloc[t0 - hv_window: t0]
            sigma_pre = float(pre_slice.std(ddof=1) * np.sqrt(ANNUAL_DAYS))
            if not np.isfinite(sigma_pre) or sigma_pre <= 0:
                skipped += 1
                if verbose:
                    print(f"[skip t0={t0}] sigma_pre 非法: {sigma_pre}")
                continue

            # 4.2 真实价格切片：保留真实价格水平，长度 T+1（含 t0 当日建仓价）
            # log_ret.index[t0] 对应 prices_series.index[t0 + 1]，
            # 因此建仓日（t0 当天）价格为 prices_arr[t0]，到期日为 prices_arr[t0 + T]。
            real_vals = prices_arr[t0: t0 + T + 1].astype(float)
            if len(real_vals) != T + 1:
                skipped += 1
                if verbose:
                    print(f"[skip t0={t0}] 真实价格切片长度 {len(real_vals)} != {T + 1}")
                continue
            real_s0 = float(real_vals[0])
            if not np.isfinite(real_s0) or real_s0 <= 0:
                skipped += 1
                if verbose:
                    print(f"[skip t0={t0}] real_s0 非法: {real_s0}")
                continue

            # 4.3 构造期权实例（先以配置 s0 = S_ref 创建，再按 ratio 缩放价格量纲字段）
            opt_params = dict(option_cfg)
            opt_params["sigma"] = sigma_pre
            opt_params["s0"] = cfg_s0  # 显式确保 S_ref
            opt_params.setdefault("r", effective_r)
            opt_params.setdefault("q", effective_q)
            opt_ref = option_class(**opt_params)

            opt, rescale_info = _rescale_option_to_real_s0(opt_ref, real_s0)
            if first_rescale_info is None:
                first_rescale_info = rescale_info

            # 4.4 MC 对照路径（独立种子，长度 T+1）；以 real_s0 为起点，
            # 与历史路径处于同一价格水平，对照公平。
            rng = np.random.default_rng(mc_seed + t0)
            drift = (effective_r - effective_q - 0.5 * sigma_pre ** 2) * dt
            diff = sigma_pre * np.sqrt(dt)
            mc_lr = rng.normal(drift, diff, size=T)
            mc_path = np.empty(T + 1)
            mc_path[0] = real_s0
            mc_path[1:] = real_s0 * np.exp(np.cumsum(mc_lr))

            # 4.5 两边回测
            bt_real = HedgeBacktest(
                option=opt,
                path_source="historical",
                external_path=real_vals,
                **hk_base,
            )
            res_real = bt_real.run()

            bt_mc = HedgeBacktest(
                option=opt,
                path_source="gbm",
                prices=mc_path,
                **hk_base,
            )
            res_mc = bt_mc.run()

            # Phase 2 保证存在 'total_pnl' 与 'delta_discretization_pnl'
            disc_real_arr = res_real.get("delta_discretization_pnl")
            disc_mc_arr = res_mc.get("delta_discretization_pnl")
            delta_disc_real = (
                float(disc_real_arr[-1]) if (is_future and disc_real_arr is not None and len(disc_real_arr)) else 0.0
            )
            delta_disc_mc = (
                float(disc_mc_arr[-1]) if (is_future and disc_mc_arr is not None and len(disc_mc_arr)) else 0.0
            )

            rows.append(
                {
                    # 建仓价是 prices_arr[t0]，建仓日就必须是它自己的日期。
                    # 旧写法取 log_ret.index[t0]，而对齐关系是
                    # log_ret.index[k] <-> prices_arr[k + 1]（见上面的说明），
                    # 于是记录下来的日期整整晚了一个交易日：real_s0 是 T 日的
                    # 收盘价，start_date 却写着 T+1 日。
                    "start_date": prices_index[t0],
                    "position": position,
                    "position_label": "sell" if position == 1 else "buy",
                    "sigma_pre": sigma_pre,
                    "real_s0": real_s0,
                    "rescale_ratio": float(rescale_info["ratio"]),
                    "hedge_pnl_real": float(res_real["total_pnl"]),
                    "hedge_pnl_mc": float(res_mc["total_pnl"]),
                    "final_price_real": float(real_vals[-1]),
                    "final_price_mc": float(mc_path[-1]),
                    "delta_disc_real": delta_disc_real,
                    "delta_disc_mc": delta_disc_mc,
                }
            )
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[skip t0={t0}] 异常: {type(e).__name__}: {e}")
            continue

    if verbose or skipped > 0:
        print(f"[rolling_backtest] rows={len(rows)}, skipped={skipped}")

    df = pd.DataFrame(rows)
    # 把首轮的期权要素伸缩明细挂到 DataFrame.attrs 上，便于 GUI 展示。
    # 注意：每一轮的 ratio 单独存在 'rescale_ratio' 列中。
    df.attrs["first_rescale_info"] = first_rescale_info
    df.attrs["s_ref"] = cfg_s0
    df.attrs["position"] = position
    df.attrs["position_label"] = "sell" if position == 1 else "buy"
    df.attrs["quantity"] = quantity
    return df
