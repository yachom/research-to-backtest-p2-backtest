"""성과지표 — README §24.1 전 항목, 공식 고정 (명세 A6 §4).

일별 수익률 ``r = equity.pct_change()``, 연환산 계수 252(거래일). 모든 지표는
분모 0·표본 부족 등 정의 불가 구간에서 **NaN이 아니라 None**을 반환한다(명세
A6 §4 엣지 케이스). 거래 0건이면 거래 기반 지표는 None이고 ``has_trades=False``.

벤치마크는 KOSPI 동일 기간 buy&hold이며, Buy & Hold 비교(전략과 동일 비용으로
첫날 매수·마지막날 보유)도 함께 산출한다(README M9 DoD "Buy & Hold 비교").
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.engine import EngineResult, TradeRecord

TRADING_DAYS_PER_YEAR = 252


class BuyHoldComparison(BaseModel):
    """대상 자산 Buy & Hold(첫날 시가 매수·마지막날 종가 보유, 진입 비용 반영)."""

    model_config = ConfigDict(extra="forbid")

    cumulative_return: float | None
    cagr: float | None
    max_drawdown: float | None


class BenchmarkComparison(BaseModel):
    """벤치마크(KOSPI) 대비 비교 (README §24.1)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cumulative_return: float | None  # 벤치마크 buy&hold 누적수익률
    excess_return: float | None  # 전략 - 벤치마크 누적수익률
    information_ratio: float | None


class BacktestResult(BaseModel):
    """백테스트 성과 요약 — 산출물 backtest_result.json의 본문 (명세 A6 §4).

    기간·설정 에코 + 성과지표 전부 + 벤치마크·B&H 비교 + ``has_trades``.
    """

    model_config = ConfigDict(extra="forbid")

    # --- 기간·설정 에코 ---
    strategy_name: str
    start_date: date
    end_date: date
    trading_days: int
    fs_scope: str
    initial_cash: float
    commission_rate: float
    sell_tax_rate: float
    slippage_rate: float

    # --- 성과지표 (README §24.1) ---
    cumulative_return: float | None
    cagr: float | None
    annual_volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    calmar: float | None
    win_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    payoff_ratio: float | None
    profit_factor: float | None
    num_trades: int
    avg_holding_days: float | None
    market_exposure: float | None

    # --- 벤치마크·B&H ---
    benchmark: BenchmarkComparison
    buy_hold: BuyHoldComparison

    has_trades: bool


def compute_backtest_metrics(
    *,
    engine_result: EngineResult,
    asset_frame: pd.DataFrame,
    benchmark_close: pd.Series,
    config: BacktestConfig,
    strategy_name: str,
    start_date: date,
    end_date: date,
    fs_scope: str,
) -> BacktestResult:
    """엔진 산출물과 벤치마크로 성과지표를 계산한다 (명세 A6 §4).

    ``asset_frame``은 [start, end] 절단된 대상 자산 프레임(open/close,
    index=date), ``benchmark_close``는 같은 거래일에 정렬된 KOSPI 종가다.
    """
    equity = engine_result.equity_series()
    n_days = len(equity)
    returns = equity.pct_change(fill_method=None).dropna()
    trades = engine_result.trades
    has_trades = len(trades) > 0

    cumulative_return = _cumulative_return(equity)
    cagr = _cagr(equity, n_days)
    annual_volatility = _annualized_std(returns)
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    max_drawdown = _max_drawdown(equity)
    calmar = _calmar(cagr, max_drawdown)

    win_rate, avg_win, avg_loss, payoff, profit_factor = _trade_stats(trades)
    avg_holding = float(np.mean([t.holding_days for t in trades])) if has_trades else None
    exposure_days = sum(1 for row in engine_result.daily if row.position == 1)
    market_exposure = exposure_days / n_days if n_days > 0 else None

    benchmark = _benchmark_comparison(config.benchmark, cumulative_return, benchmark_close, returns)
    buy_hold = _buy_hold_comparison(asset_frame, config, n_days)

    return BacktestResult(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        trading_days=n_days,
        fs_scope=fs_scope,
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        sell_tax_rate=config.sell_tax_rate,
        slippage_rate=config.slippage_rate,
        cumulative_return=cumulative_return,
        cagr=cagr,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff,
        profit_factor=profit_factor,
        num_trades=len(trades),
        avg_holding_days=avg_holding,
        market_exposure=market_exposure,
        benchmark=benchmark,
        buy_hold=buy_hold,
        has_trades=has_trades,
    )


# --- 개별 지표 ---------------------------------------------------------------


def _finite(value: float) -> float | None:
    """NaN·inf를 None으로 흡수한다 (명세 A6 §4: NaN 금지)."""
    return float(value) if math.isfinite(value) else None


def _cumulative_return(equity: pd.Series) -> float | None:
    if len(equity) == 0 or equity.iloc[0] == 0:
        return None
    return _finite(equity.iloc[-1] / equity.iloc[0] - 1.0)


def _cagr(equity: pd.Series, n_days: int) -> float | None:
    if n_days <= 0 or len(equity) == 0 or equity.iloc[0] <= 0:
        return None
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return None
    return _finite(ratio ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)


def _annualized_std(returns: pd.Series) -> float | None:
    if len(returns) < 2:
        return None
    return _finite(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe(returns: pd.Series) -> float | None:
    """mean(r)/std(r)*√252, rf=0. std=0·표본부족 → None."""
    if len(returns) < 2:
        return None
    std = returns.std(ddof=1)
    if std == 0 or not math.isfinite(std):
        return None
    return _finite(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(returns: pd.Series) -> float | None:
    """mean(r)/std(r[r<0])*√252. 음수 수익률 부족·하방표준편차 0 → None."""
    if len(returns) < 2:
        return None
    downside = returns[returns < 0]
    if len(downside) < 2:
        return None
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or not math.isfinite(downside_std):
        return None
    return _finite(returns.mean() / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: pd.Series) -> float | None:
    if len(equity) == 0:
        return None
    drawdown = equity / equity.cummax() - 1.0
    return _finite(drawdown.min())


def _calmar(cagr: float | None, max_drawdown: float | None) -> float | None:
    if cagr is None or max_drawdown is None or max_drawdown == 0:
        return None
    return _finite(cagr / abs(max_drawdown))


def _trade_stats(
    trades: list[TradeRecord],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """승률·평균손익·payoff·profit factor (거래 0건이면 전부 None)."""
    if not trades:
        return None, None, None, None, None
    pnls = np.array([t.pnl for t in trades], dtype="float64")
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = _finite(len(wins) / len(pnls))
    avg_win = _finite(float(wins.mean())) if len(wins) > 0 else None
    avg_loss = _finite(float(losses.mean())) if len(losses) > 0 else None
    payoff = (
        _finite(avg_win / abs(avg_loss))
        if avg_win is not None and avg_loss is not None and avg_loss != 0
        else None
    )
    loss_sum = float(losses.sum())
    profit_factor = (
        _finite(float(wins.sum()) / abs(loss_sum)) if len(losses) > 0 and loss_sum != 0 else None
    )
    return win_rate, avg_win, avg_loss, payoff, profit_factor


def _benchmark_comparison(
    name: str,
    strategy_cum: float | None,
    benchmark_close: pd.Series,
    strategy_returns: pd.Series,
) -> BenchmarkComparison:
    """KOSPI buy&hold 누적수익률·초과수익률·Information Ratio (README §24.1)."""
    bench = benchmark_close.dropna()
    bench_cum: float | None = None
    if len(bench) >= 2 and bench.iloc[0] != 0:
        bench_cum = _finite(bench.iloc[-1] / bench.iloc[0] - 1.0)

    excess: float | None = None
    if strategy_cum is not None and bench_cum is not None:
        excess = _finite(strategy_cum - bench_cum)

    information_ratio = _information_ratio(strategy_returns, benchmark_close)
    return BenchmarkComparison(
        name=name,
        cumulative_return=bench_cum,
        excess_return=excess,
        information_ratio=information_ratio,
    )


def _information_ratio(strategy_returns: pd.Series, benchmark_close: pd.Series) -> float | None:
    """mean(r-r_bm)/std(r-r_bm)*√252. 표본부족·std=0 → None."""
    bench_returns = benchmark_close.pct_change(fill_method=None)
    active = (strategy_returns - bench_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(active) < 2:
        return None
    std = active.std(ddof=1)
    if std == 0 or not math.isfinite(std):
        return None
    return _finite(active.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def _buy_hold_comparison(
    asset_frame: pd.DataFrame, config: BacktestConfig, n_days: int
) -> BuyHoldComparison:
    """대상 자산 Buy & Hold — 첫날 시가 매수(슬리피지+수수료), 마지막날 종가 보유."""
    if asset_frame.empty:
        return BuyHoldComparison(cumulative_return=None, cagr=None, max_drawdown=None)

    opens = asset_frame["open"].astype("float64").to_numpy()
    closes = asset_frame["close"].astype("float64").to_numpy()
    initial_cash = float(config.initial_cash)

    fill_price = opens[0] * (1.0 + config.slippage_rate)
    shares = math.floor(initial_cash / fill_price) if fill_price > 0 else 0
    if shares == 0:
        return BuyHoldComparison(cumulative_return=None, cagr=None, max_drawdown=None)
    gross = shares * fill_price
    commission = gross * config.commission_rate
    cash_residual = initial_cash - gross - commission

    equity_bh = pd.Series(cash_residual + shares * closes, index=asset_frame.index)
    cum = _finite(equity_bh.iloc[-1] / initial_cash - 1.0)
    cagr: float | None = None
    if n_days > 0:
        ratio = equity_bh.iloc[-1] / initial_cash
        if ratio > 0:
            cagr = _finite(ratio ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)
    mdd = _max_drawdown(equity_bh)
    return BuyHoldComparison(cumulative_return=cum, cagr=cagr, max_drawdown=mdd)
