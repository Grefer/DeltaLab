# _*_ coding: utf-8 _*_
"""
期权动态对冲回测框架

支持对任意继承自 OptionBase 的期权进行 Delta 对冲回测，
追踪每日盈亏分解、累计对冲误差和 Greeks 变动。
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

try:
    from .constants import ANNUAL_DAYS
except ImportError:
    from constants import ANNUAL_DAYS


class HedgeBacktest:
    """
    期权动态对冲回测

    Parameters
    ----------
    option : OptionBase
        初始期权对象（t=0 时刻状态）
    prices : array-like
        标的资产每日价格序列，prices[0] = 期初价格，长度 = 总交易日数 + 1
    hedge_freq : int
        调仓频率（每隔几个交易日调仓），默认 1（每日调仓）
    tc_rate : float
        单边交易成本率，默认 0
    position : int
        期权头寸方向：1 = 卖出（short）期权，-1 = 买入（long）期权
    quantity : float
        交易数量（如 10 吨、1000 股等），用于将 Delta 转换为实际持仓。
        实际持仓 = position * delta * quantity。默认 1（等价于 1 份标的）
    multiplier : float
        合约乘数（每手对应的标的数量），用于取整到整数手。
        手数 = round(持仓 / multiplier)，实际持仓 = 手数 * multiplier。
        0 表示不取整（连续对冲）。默认 0。
        例：quantity=10, multiplier=10 → 最多 1 手。

    Examples
    --------
    >>> from pricing import Option_AB, HedgeBacktest
    >>> opt = Option_AB('Opt_Airbag', 100, [], 100, 90, 20, list(range(1,21)),
    ...                 0.18, 0.8, 1, 1)
    >>> prices = HedgeBacktest.simulate_prices(100, 0.18, 20)
    >>> bt = HedgeBacktest(opt, prices, hedge_freq=1, tc_rate=0.001, position=1,
    ...                    quantity=10, multiplier=10)
    >>> results = bt.run()
    >>> bt.summary()
    >>> bt.plot()
    """

    def __init__(self, option, prices, hedge_freq=1, tc_rate=0.0, position=1,
                 quantity=1.0, multiplier=0):
        self.option_init = copy.deepcopy(option)
        self.prices = np.asarray(prices, dtype=float)
        self.hedge_freq = max(1, int(hedge_freq))
        self.tc_rate = tc_rate
        self.position = position
        self.quantity = float(quantity)
        self.multiplier = float(multiplier)
        self._results = None

    def run(self):
        """执行回测，返回结果字典"""
        option = copy.deepcopy(self.option_init)
        S = self.prices
        n = len(S) - 1
        r = option.r
        dt = 1.0 / ANNUAL_DAYS
        pos = self.position

        # 存储数组
        V = np.zeros(n + 1)        # 期权理论价值
        delta = np.zeros(n + 1)    # Delta
        gamma = np.zeros(n + 1)    # Gamma
        vega = np.zeros(n + 1)     # Vega
        theta = np.zeros(n + 1)    # Theta
        rho = np.zeros(n + 1)      # Rho
        H = np.zeros(n + 1)        # 持有标的数量
        C = np.zeros(n + 1)        # 现金账户
        TC = np.zeros(n + 1)       # 当日交易成本

        # 取整辅助函数
        qty = self.quantity
        mult = self.multiplier

        def _round_to_lots(target):
            """将目标持仓取整到 multiplier 的整数倍（向零取整）"""
            if mult <= 0:
                return target
            if target >= 0:
                return int(target / mult) * mult
            else:
                return -int(-target / mult) * mult

        def _target_shares(delta_val):
            """根据 Delta 计算目标持仓（含数量缩放与取整）"""
            raw = pos * delta_val * qty
            return _round_to_lots(raw)

        # ---- Day 0: 建仓 ----
        V[0] = option.get_price() or 0.0
        greeks = option.get_greeks()
        delta[0], gamma[0], vega[0], theta[0], rho[0] = greeks

        H[0] = _target_shares(delta[0])
        TC[0] = abs(H[0]) * S[0] * self.tc_rate
        C[0] = pos * V[0] * qty - H[0] * S[0] - TC[0]

        # ---- Day 1 ~ n ----
        for i in range(1, n + 1):
            # 推进一天
            option.step_forward(S[i])
            time_left = option._time_remaining

            # 计算价值与 Greeks
            if time_left > 0:
                val = option.get_price()
                V[i] = val if val is not None else 0.0
                greeks = option.get_greeks()
                delta[i], gamma[i], vega[i], theta[i], rho[i] = greeks
            elif time_left == 0:
                val = option.get_price()
                V[i] = val if val is not None else 0.0
                delta[i] = gamma[i] = vega[i] = theta[i] = rho[i] = 0.0
            else:
                V[i] = delta[i] = gamma[i] = vega[i] = theta[i] = rho[i] = 0.0

            # 现金隔夜利息
            C[i] = C[i - 1] * np.exp(r * dt)
            H[i] = H[i - 1]

            # 调仓逻辑
            if i == n:
                # 到期日：平仓所有标的头寸
                trade = -H[i]
                TC[i] = abs(trade) * S[i] * self.tc_rate
                C[i] -= trade * S[i] + TC[i]
                H[i] = 0.0
            elif i % self.hedge_freq == 0:
                target = _target_shares(delta[i])
                trade = target - H[i]
                TC[i] = abs(trade) * S[i] * self.tc_rate
                C[i] -= trade * S[i] + TC[i]
                H[i] = target

        # ---- 盈亏分解 ----
        PV = C + H * S  # 组合市值（现金 + 标的持仓）

        hedge_daily = np.zeros(n + 1)     # 标的头寸每日盈亏
        option_daily = np.zeros(n + 1)    # 期权 MtM 每日盈亏（从对冲方视角）
        interest_daily = np.zeros(n + 1)  # 利息收入

        for i in range(1, n + 1):
            hedge_daily[i] = H[i - 1] * (S[i] - S[i - 1])
            option_daily[i] = -pos * (V[i] - V[i - 1]) * qty
            interest_daily[i] = C[i - 1] * (np.exp(r * dt) - 1)

        net_daily = hedge_daily + option_daily + interest_daily - TC
        cum_pnl = np.cumsum(net_daily)

        # 最终对冲误差：完美对冲时应接近 0
        hedging_error = PV[-1] - pos * V[-1] * qty

        self._results = {
            'n_days': n,
            'prices': S,
            'opt_value': V,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'shares': H,
            'cash': C,
            'portfolio_val': PV,
            'hedge_daily': hedge_daily,
            'option_daily': option_daily,
            'interest_daily': interest_daily,
            'tc_paid': TC,
            'net_daily': net_daily,
            'cumulative_pnl': cum_pnl,
            'hedging_error': hedging_error,
            'total_tc': np.sum(TC),
        }
        return self._results

    def summary(self):
        """打印回测摘要"""
        if self._results is None:
            self.run()
        r = self._results
        n = r['n_days']

        lines = [
            "=" * 52,
            "          动态对冲回测结果摘要",
            "=" * 52,
            f"  回测天数          :  {n:>10d}",
            f"  调仓频率          :  每 {self.hedge_freq} 天",
            f"  交易成本率        :  {self.tc_rate * 100:.2f}%",
            "-" * 52,
            f"  标的初始价格      :  {r['prices'][0]:>12.4f}",
            f"  标的到期价格      :  {r['prices'][-1]:>12.4f}",
            f"  标的涨跌幅        :  {(r['prices'][-1] / r['prices'][0] - 1) * 100:>11.2f}%",
            "-" * 52,
            f"  期权初始价值      :  {r['opt_value'][0]:>12.4f}",
            f"  期权到期价值      :  {r['opt_value'][-1]:>12.4f}",
            "-" * 52,
            f"  标的对冲盈亏      :  {np.sum(r['hedge_daily']):>12.4f}",
            f"  期权 MtM 盈亏     :  {np.sum(r['option_daily']):>12.4f}",
            f"  利息收入          :  {np.sum(r['interest_daily']):>12.4f}",
            f"  累计交易成本      :  {r['total_tc']:>12.4f}",
            f"  对冲误差          :  {r['hedging_error']:>12.4f}",
            "-" * 52,
            "  【Greeks 统计】",
            f"  {'':15s}  {'初始值':>10s}  {'均值':>10s}  {'最大|值|':>10s}",
            f"  {'Delta':15s}  {r['delta'][0]:>10.4f}  {np.mean(r['delta']):>10.4f}  {np.max(np.abs(r['delta'])):>10.4f}",
            f"  {'Gamma':15s}  {r['gamma'][0]:>10.4f}  {np.mean(r['gamma']):>10.4f}  {np.max(np.abs(r['gamma'])):>10.4f}",
            f"  {'Vega':15s}  {r['vega'][0]:>10.4f}  {np.mean(r['vega']):>10.4f}  {np.max(np.abs(r['vega'])):>10.4f}",
            f"  {'Theta':15s}  {r['theta'][0]:>10.4f}  {np.mean(r['theta']):>10.4f}  {np.max(np.abs(r['theta'])):>10.4f}",
            f"  {'Rho':15s}  {r['rho'][0]:>10.4f}  {np.mean(r['rho']):>10.4f}  {np.max(np.abs(r['rho'])):>10.4f}",
            "-" * 52,
            f"  调仓次数          :  {int(np.sum(np.abs(np.diff(r['shares'])) > 1e-10)):>10d}",
            "=" * 52,
        ]
        print("\n".join(lines))
        return r

    def plot(self, figsize=(14, 10)):
        """绘制回测结果四宫格图"""
        if self._results is None:
            self.run()
        r = self._results
        days = np.arange(len(r['prices']))

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # (1) 标的价格走势
        ax = axes[0, 0]
        ax.plot(days, r['prices'], 'b-', linewidth=1.2)
        ax.set_title('标的价格路径', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('价格')
        ax.grid(True, alpha=0.3)

        # (2) Delta 与实际持仓
        ax = axes[0, 1]
        ax.plot(days, r['delta'], 'r-', label='Delta', linewidth=1.2)
        ax.plot(days, r['shares'] / self.position if self.position != 0 else r['shares'],
                'b--', label='实际持仓/pos', linewidth=1.0, alpha=0.7)
        ax.set_title('Delta 与对冲持仓', fontsize=12)
        ax.set_xlabel('交易日')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (3) 期权理论价值
        ax = axes[1, 0]
        ax.plot(days, r['opt_value'], 'g-', linewidth=1.2)
        ax.set_title('期权理论价值', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('价值')
        ax.grid(True, alpha=0.3)

        # (4) 累计对冲盈亏
        ax = axes[1, 1]
        cp = r['cumulative_pnl']
        ax.plot(days, cp, 'k-', linewidth=1.2, label='累计盈亏')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(days, 0, cp, where=cp >= 0, alpha=0.15, color='green')
        ax.fill_between(days, 0, cp, where=cp < 0, alpha=0.15, color='red')
        ax.set_title('累计对冲盈亏', fontsize=12)
        ax.set_xlabel('交易日')
        ax.set_ylabel('盈亏')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

    def to_dataframe(self):
        """将回测结果转为 pandas DataFrame"""
        if self._results is None:
            self.run()
        r = self._results
        import pandas as pd
        df = pd.DataFrame({
            '标的价格': r['prices'],
            '期权价值': r['opt_value'],
            'Delta': r['delta'],
            'Gamma': r['gamma'],
            'Vega': r['vega'],
            'Theta': r['theta'],
            'Rho': r['rho'],
            '持仓': r['shares'],
            '现金': r['cash'],
            '组合市值': r['portfolio_val'],
            '标的盈亏': r['hedge_daily'],
            '期权盈亏': r['option_daily'],
            '利息': r['interest_daily'],
            '交易成本': r['tc_paid'],
            '每日净盈亏': r['net_daily'],
            '累计盈亏': r['cumulative_pnl'],
        })
        # 使用 Wind 日期作为索引（如有）
        if hasattr(self, '_wind_meta') and self._wind_meta is not None:
            df.index = self._wind_meta['dates']
            df.index.name = '日期'
        else:
            df.index.name = '交易日'
        return df

    @staticmethod
    def simulate_prices(s0, sigma, T_days, r=0.03, q=0.03, seed=42):
        """
        生成 GBM 模拟价格路径，用于回测测试

        Parameters
        ----------
        s0 : float
            初始价格
        sigma : float
            波动率
        T_days : int
            总交易日数
        r, q : float
            无风险利率、分红率
        seed : int
            随机种子

        Returns
        -------
        prices : ndarray, shape (T_days + 1,)
        """
        rng = np.random.default_rng(seed)
        dt = 1.0 / ANNUAL_DAYS
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)
        log_returns = drift + vol * rng.standard_normal(T_days)
        prices = np.zeros(T_days + 1)
        prices[0] = s0
        prices[1:] = s0 * np.exp(np.cumsum(log_returns))
        return prices

    @staticmethod
    def simulate_multi_paths(s0, sigma, T_days, n_paths=100, r=0.03, q=0.03, seed=42):
        """
        批量生成多条价格路径，用于对冲效果的统计分析

        Returns
        -------
        paths : ndarray, shape (n_paths, T_days + 1)
        """
        rng = np.random.default_rng(seed)
        dt = 1.0 / ANNUAL_DAYS
        drift = (r - q - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)
        log_returns = drift + vol * rng.standard_normal((n_paths, T_days))
        paths = np.zeros((n_paths, T_days + 1))
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(np.cumsum(log_returns, axis=1))
        return paths

    def run_multi(self, paths):
        """
        对多条价格路径批量回测，返回对冲误差分布

        Parameters
        ----------
        paths : ndarray, shape (n_paths, T_days + 1)

        Returns
        -------
        errors : ndarray, shape (n_paths,)
            每条路径的对冲误差
        """
        errors = np.zeros(len(paths))
        for i, p in enumerate(paths):
            bt = HedgeBacktest(
                self.option_init, p,
                hedge_freq=self.hedge_freq,
                tc_rate=self.tc_rate,
                position=self.position,
                quantity=self.quantity,
                multiplier=self.multiplier,
            )
            res = bt.run()
            errors[i] = res['hedging_error']
        return errors

    def plot_error_dist(self, errors, figsize=(10, 5)):
        """绘制对冲误差分布直方图"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(errors), color='red', linestyle='--',
                   label=f'均值={np.mean(errors):.4f}')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(f'对冲误差分布 (n={len(errors)}, std={np.std(errors):.4f})', fontsize=12)
        ax.set_xlabel('对冲误差')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    # ============================================================
    #  Wind 数据接入
    # ============================================================

    @classmethod
    def from_wind(cls, option, code, start_date, end_date,
                  hedge_freq=1, tc_rate=0.0, position=1, adjust="F",
                  quantity=1.0, multiplier=0):
        """
        使用 Wind 历史行情创建回测实例

        Parameters
        ----------
        option : OptionBase
            期权对象（s0 会被替换为 start_date 当日收盘价）
        code : str
            Wind 标的代码，如 "000001.SH", "510050.SH"
        start_date : str
            回测起始日 "YYYY-MM-DD"（含当日，作为建仓日）
        end_date : str
            回测结束日 "YYYY-MM-DD"（含当日）
        hedge_freq : int
            调仓频率（交易日数），默认 1
        tc_rate : float
            单边交易成本率
        position : int
            1=卖出期权, -1=买入期权
        adjust : str
            复权方式 "F"=前复权, "B"=后复权, ""=不复权

        Returns
        -------
        HedgeBacktest
            已设置好价格序列的回测实例

        Examples
        --------
        >>> from pricing import Option_AS, HedgeBacktest
        >>> opt = Option_AS('Asian', 0, [], 100, 100, 22, 22, 0.15, 1, 0, float('inf'))
        >>> bt = HedgeBacktest.from_wind(opt, '510050.SH', '2025-01-02', '2025-02-07')
        >>> bt.run()
        >>> bt.summary()
        """
        try:
            from .wind_data import get_close_prices
        except ImportError:
            from wind_data import get_close_prices

        series = get_close_prices(code, start_date, end_date, adjust)
        prices = series.values

        if len(prices) < 2:
            raise ValueError(f"价格数据不足: {code} {start_date}~{end_date} 仅 {len(prices)} 条")

        # 用实际起始价替换 option.s0
        opt = copy.deepcopy(option)
        opt.s0 = float(prices[0])

        bt = cls(opt, prices, hedge_freq=hedge_freq, tc_rate=tc_rate, position=position,
                 notional=notional, lot_size=lot_size)
        bt._wind_meta = {
            'code': code,
            'start_date': start_date,
            'end_date': end_date,
            'dates': series.index,
            'n_trade_days': len(prices) - 1,
        }
        return bt

    @classmethod
    def from_csv(cls, option, filepath, price_col='close',
                 date_col=None, hedge_freq=1, tc_rate=0.0, position=1,
                 quantity=1.0, multiplier=0):
        """
        从 CSV 文件加载价格数据创建回测实例（无需 Wind 终端）

        Parameters
        ----------
        filepath : str
            CSV 文件路径
        price_col : str
            收盘价列名，默认 'close'
        date_col : str or None
            日期列名，None 则使用第一列

        Returns
        -------
        HedgeBacktest
        """
        import pandas as pd
        if date_col:
            df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)
        else:
            df = pd.read_csv(filepath, parse_dates=[0], index_col=0)

        if price_col not in df.columns:
            raise ValueError(f"列 '{price_col}' 不在 CSV 中，可用列: {list(df.columns)}")

        series = df[price_col].dropna()
        prices = series.values

        opt = copy.deepcopy(option)
        opt.s0 = float(prices[0])

        bt = cls(opt, prices, hedge_freq=hedge_freq, tc_rate=tc_rate, position=position,
                 quantity=quantity, multiplier=multiplier)
        bt._wind_meta = {
            'code': filepath,
            'start_date': str(series.index[0].date()),
            'end_date': str(series.index[-1].date()),
            'dates': series.index,
            'n_trade_days': len(prices) - 1,
        }
        return bt
