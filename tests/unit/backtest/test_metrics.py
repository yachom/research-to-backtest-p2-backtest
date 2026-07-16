"""metrics.py — 성과지표 손계산 대조·엣지 None·B&H 비교 (명세 A6 §4·§6, README §24.1)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.engine import (
    DailyPortfolioRow,
    EngineResult,
    ExitReason,
    TradeRecord,
)
from research_backtest.quant.backtest.metrics import TRADING_DAYS_PER_YEAR, compute_backtest_metrics

ZERO_COST = BacktestConfig(
    commission_rate=0.0, sell_tax_rate=0.0, slippage_rate=0.0, initial_cash=100.0
)


def _days(n: int) -> list[date]:
    base = date(2024, 1, 1)
    out: list[date] = []
    cursor = base
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _engine_result(
    equity: Sequence[float],
    positions: Sequence[int] | None = None,
    trades: Sequence[TradeRecord] = (),
) -> EngineResult:
    dates = _days(len(equity))
    pos = positions if positions is not None else [0] * len(equity)
    rows = [
        DailyPortfolioRow(
            date=dates[i],
            position=pos[i],
            shares=0,
            cash=equity[i],
            equity=equity[i],
            daily_return=0.0,
        )
        for i in range(len(equity))
    ]
    return EngineResult(trades=list(trades), daily=rows)


def _trade(pnl: float, holding: int = 5) -> TradeRecord:
    return TradeRecord(
        entry_signal_date=date(2024, 1, 1),
        entry_date=date(2024, 1, 2),
        entry_price=100.0,
        shares=10,
        exit_signal_date=date(2024, 1, 8),
        exit_date=date(2024, 1, 9),
        exit_price=100.0 + pnl / 10,
        holding_days=holding,
        pnl=pnl,
        pnl_pct=pnl / 1000.0,
        exit_reason=ExitReason.SIGNAL,
        costs=0.0,
    )


def _flat_asset(dates: Sequence[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [100.0] * len(dates), "close": [100.0] * len(dates)},
        index=pd.Index(dates, name="date"),
    )


def _metrics(engine_result: EngineResult, benchmark: Sequence[float] | None = None):  # type: ignore[no-untyped-def]
    dates = [row.date for row in engine_result.daily]
    asset = pd.DataFrame(
        {
            "open": [row.equity for row in engine_result.daily],
            "close": [row.equity for row in engine_result.daily],
        },
        index=pd.Index(dates, name="date"),
    )
    bench = pd.Series(
        list(benchmark) if benchmark is not None else [float("nan")] * len(dates),
        index=pd.Index(dates, name="date"),
    )
    return compute_backtest_metrics(
        engine_result=engine_result,
        asset_frame=asset,
        benchmark_close=bench,
        config=ZERO_COST,
        strategy_name="T",
        start_date=dates[0],
        end_date=dates[-1],
        fs_scope="CFS",
    )


# --- 누적수익률·CAGR·MDD 손계산 ----------------------------------------------


def test_cumulative_cagr_mdd_hand_calc() -> None:
    equity = [100.0, 110.0, 99.0, 108.0]
    result = _metrics(_engine_result(equity))
    assert result.cumulative_return == pytest.approx(0.08)  # 108/100 - 1
    assert result.cagr == pytest.approx(1.08 ** (TRADING_DAYS_PER_YEAR / 4) - 1)
    # cummax=[100,110,110,110] → drawdown min = 99/110-1
    assert result.max_drawdown == pytest.approx(99 / 110 - 1)
    assert result.calmar == pytest.approx(result.cagr / abs(result.max_drawdown))


def test_annual_volatility_and_sharpe_hand_calc() -> None:
    equity = [100.0, 110.0, 99.0, 108.0]
    returns = pd.Series(equity).pct_change().dropna()
    result = _metrics(_engine_result(equity))
    assert result.annual_volatility == pytest.approx(
        returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    assert result.sharpe == pytest.approx(
        returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    )


# --- 거래 기반 지표 손계산 ----------------------------------------------------


def test_trade_stats_hand_calc() -> None:
    trades = [_trade(100.0, 4), _trade(-50.0, 6), _trade(30.0, 8)]
    result = _metrics(_engine_result([100.0, 100.0], trades=trades))
    assert result.num_trades == 3
    assert result.has_trades is True
    assert result.win_rate == pytest.approx(2 / 3)  # 2승 1패
    assert result.avg_win == pytest.approx(65.0)  # (100+30)/2
    assert result.avg_loss == pytest.approx(-50.0)
    assert result.payoff_ratio == pytest.approx(65.0 / 50.0)
    assert result.profit_factor == pytest.approx(130.0 / 50.0)
    assert result.avg_holding_days == pytest.approx((4 + 6 + 8) / 3)


def test_market_exposure() -> None:
    result = _metrics(_engine_result([100.0] * 4, positions=[0, 1, 1, 0]))
    assert result.market_exposure == pytest.approx(0.5)  # 2/4


# --- 엣지 케이스: None (명세 A6 §4) ------------------------------------------


def test_zero_trades_metrics_are_none() -> None:
    result = _metrics(_engine_result([100.0, 100.0, 100.0]))
    assert result.has_trades is False
    assert result.num_trades == 0
    assert result.win_rate is None
    assert result.avg_win is None
    assert result.avg_loss is None
    assert result.payoff_ratio is None
    assert result.profit_factor is None
    assert result.avg_holding_days is None


def test_flat_equity_sharpe_none_but_vol_zero() -> None:
    """std=0(횡보) → Sharpe·Sortino None, 변동성은 0.0(정의됨)."""
    result = _metrics(_engine_result([100.0, 100.0, 100.0, 100.0]))
    assert result.sharpe is None
    assert result.sortino is None
    assert result.annual_volatility == pytest.approx(0.0)
    assert result.calmar is None  # MDD=0 → 분모 0
    assert result.max_drawdown == pytest.approx(0.0)


def test_no_losing_trades_profit_factor_none() -> None:
    """손실 거래가 없으면 profit_factor·payoff None(분모 0)."""
    result = _metrics(_engine_result([100.0, 100.0], trades=[_trade(100.0), _trade(50.0)]))
    assert result.profit_factor is None
    assert result.payoff_ratio is None
    assert result.avg_loss is None
    assert result.win_rate == pytest.approx(1.0)


def test_all_metrics_finite_or_none_never_nan() -> None:
    """NaN 금지 — 모든 성과지표는 float이거나 None."""
    result = _metrics(_engine_result([100.0, 100.0]))
    for name in (
        "cumulative_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "profit_factor",
    ):
        value = getattr(result, name)
        assert value is None or math.isfinite(value)


# --- 벤치마크·B&H 비교 --------------------------------------------------------


def test_benchmark_excess_and_information_ratio() -> None:
    equity = [100.0, 110.0, 121.0]  # 전략 +21%
    benchmark = [100.0, 105.0, 110.0]  # 벤치 +10%
    result = _metrics(_engine_result(equity), benchmark=benchmark)
    assert result.benchmark.name == "KOSPI"
    assert result.benchmark.cumulative_return == pytest.approx(0.10)
    assert result.benchmark.excess_return == pytest.approx(0.21 - 0.10)
    assert result.benchmark.information_ratio is not None  # 활성수익률 존재


def test_buy_hold_comparison_hand_calc() -> None:
    """B&H — 첫날 시가 매수(비용 반영)·마지막날 종가 보유."""
    dates = _days(3)
    asset = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0], "close": [100.0, 120.0, 150.0]},
        index=pd.Index(dates, name="date"),
    )
    engine_result = _engine_result([100.0, 100.0, 100.0])  # 전략 무거래
    config = BacktestConfig(
        commission_rate=0.001, sell_tax_rate=0.0, slippage_rate=0.0, initial_cash=1_000_000.0
    )
    bench = pd.Series([float("nan")] * 3, index=pd.Index(dates, name="date"))
    result = compute_backtest_metrics(
        engine_result=engine_result,
        asset_frame=asset,
        benchmark_close=bench,
        config=config,
        strategy_name="T",
        start_date=dates[0],
        end_date=dates[-1],
        fs_scope="CFS",
    )
    # fill=100, shares=floor(1e6/100)=10000, commission=10000*100*0.001=1000
    # residual = 1e6 - 1e6 - 1000 = -1000; 최종 equity = -1000 + 10000*150 = 1_499_000
    assert result.buy_hold.cumulative_return == pytest.approx(1_499_000 / 1_000_000 - 1)
    assert result.buy_hold.max_drawdown == pytest.approx(0.0)  # 단조 상승


def test_missing_benchmark_yields_none() -> None:
    """벤치마크 데이터 부재(전 구간 NaN) → 벤치마크 지표 None."""
    result = _metrics(_engine_result([100.0, 110.0, 120.0]), benchmark=None)
    assert result.benchmark.cumulative_return is None
    assert result.benchmark.excess_return is None
    assert result.benchmark.information_ratio is None
