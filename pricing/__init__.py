# _*_ coding: utf-8 _*_
from .constants import ANNUAL_DAYS, CALENDAR_DAYS
from .mc_engine import McGbmQ
from .option_base import OptionBase
from .Option_Vanilla import Option_Vanilla, blsprice
from .Option_AB import Option_AB
from .Option_AS import Option_AS
from .Option_DE import Option_DE
from .Option_SNB import Option_SNB
from .hedge_backtest import (
    HedgeBacktest,
    HedgeStrategy,
    FixedFreqStrategy,
    CloseToCloseStrategy,
    FixedTimeStrategy,
    PriceIntervalStrategy,
    SigmaBandStrategy,
    HedgeBandStrategy,
)
from .hedge_analysis import (
    StrategyCase,
    ContractHistoryPool,
    compare_strategies,
    history_window_summary,
    result_daily_frame,
    summarize_strategy_result,
    recommend_by_lookback,
    recommend_by_rolling_history,
    recommend_by_contract_history_pool,
    LOOKBACK_DAYS,
    HISTORY_SELECTION_METRIC,
    HISTORY_TARGET_ENDPOINTS,
)

# Wind 数据接口（仅在安装了 WindPy 时可用）
try:
    from . import wind_data
except ImportError:
    pass
