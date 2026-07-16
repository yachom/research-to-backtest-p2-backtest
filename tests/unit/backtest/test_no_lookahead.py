"""절단 불변 property — 룩어헤드 방지 총괄 (명세 A6 §6, README §28.3).

**핵심 property**: 데이터를 뒤에서 절단해도 절단 전 구간의 거래 목록이 동일하다.
전략 신호가 t까지 정보만 쓰고(A5) 엔진이 인과적(t+1 체결)이면, 미래 데이터를
잘라내도 과거 거래는 바뀌지 않는다 — 이 불변식이 깨지면 어딘가에서 미래를
훔쳐본 것이다.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.data import build_backtest_frame, truncate_to_window
from research_backtest.quant.backtest.engine import EngineResult, run_backtest
from research_backtest.quant.strategy.compiler import compile_strategy, entry_signal, exit_signal
from research_backtest.quant.strategy.schema import parse_strategy_spec

STRATEGY: dict[str, object] = {
    "strategy_name": "PropertyWalk",
    "universe": {"type": "single_asset", "tickers": ["000660"]},
    "entry": {"all": [{"left": "close", "operator": ">", "right": "sma_5"}]},
    "exit": {
        "any": [
            {"left": "close", "operator": "cross_below", "right": "sma_5"},
            {"type": "max_holding_days", "value": 4},
            {"type": "stop_loss", "value": -0.05},
        ]
    },
}

CONFIG = BacktestConfig(
    commission_rate=0.00015, sell_tax_rate=0.0018, slippage_rate=0.001, initial_cash=10_000_000.0
)
EMPTY_METRICS = pd.DataFrame(
    columns=["metric_id", "fs_scope", "available_from", "value", "rcept_dt"]
)


def _walk_daily(n: int) -> pd.DataFrame:
    """결정적 난수 보행 종가로 진입·청산이 반복되는 daily를 만든다."""
    rng = np.random.default_rng(20260714)
    steps = rng.normal(0, 2.0, size=n).cumsum()
    closes = 1000 + steps + 15 * np.sin(np.arange(n) / 3.0)
    closes = np.maximum(closes, 100.0)
    opens = closes - rng.normal(0, 1.0, size=n)
    dates: list[date] = []
    cursor = date(2020, 1, 1)
    while len(dates) < n:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor += timedelta(days=1)
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, closes) + 1,
            "low": np.minimum(opens, closes) - 1,
            "close": closes,
            "volume": [1000] * n,
            "foreign_net_buy_value": [0] * n,
            "institution_net_buy_value": [0] * n,
        }
    )


def _run(daily: pd.DataFrame, start: date, end: date) -> EngineResult:
    """runner와 동일한 순서(join → indicators → signal → 절단 → 엔진)를 인메모리로 실행."""
    compiled = compile_strategy(parse_strategy_spec(STRATEGY))
    joined = build_backtest_frame(
        daily, EMPTY_METRICS, fs_scope="CFS", start_date=start, end_date=end
    )
    from research_backtest.quant.strategy.indicators import compute_indicators

    with_ind = compute_indicators(joined, compiled.required_columns)
    entry = entry_signal(compiled, with_ind)
    exit_ = exit_signal(compiled, with_ind)
    frame = truncate_to_window(with_ind, start, end)
    return run_backtest(
        frame,
        entry.reindex(frame.index),
        exit_.reindex(frame.index),
        compiled.position_rules,
        CONFIG,
    )


def test_backward_truncation_preserves_earlier_trades() -> None:
    """데이터를 뒤에서 잘라도 절단 전에 완결된 거래는 완전히 동일하다."""
    daily = _walk_daily(120)
    start = date(2020, 1, 1)
    full_end = daily["date"].iloc[-1]
    cut = daily["date"].iloc[80]  # 뒤 40여 행을 잘라낼 경계

    trades_full = _run(daily, start, full_end).trades
    trades_short = _run(daily.iloc[:81].copy(), start, cut).trades

    # cut 이전에 청산이 끝난 거래만 비교(경계에서 강제 END_OF_DATA되는 미결 거래 제외)
    before_full = [t.model_dump() for t in trades_full if t.exit_date < cut]
    before_short = [t.model_dump() for t in trades_short if t.exit_date < cut]

    assert len(before_full) >= 3  # property가 의미있으려면 거래가 여러 건
    assert before_full == before_short


def test_truncation_invariance_multiple_cuts() -> None:
    """여러 절단 지점에서 모두 불변식이 성립한다."""
    daily = _walk_daily(150)
    start = date(2020, 1, 1)
    full = _run(daily, start, daily["date"].iloc[-1]).trades

    for cut_idx in (60, 90, 120):
        cut = daily["date"].iloc[cut_idx]
        short = _run(daily.iloc[: cut_idx + 1].copy(), start, cut).trades
        before_full = [t.model_dump() for t in full if t.exit_date < cut]
        before_short = [t.model_dump() for t in short if t.exit_date < cut]
        assert before_full == before_short, f"cut_idx={cut_idx}에서 절단 불변 위반"
