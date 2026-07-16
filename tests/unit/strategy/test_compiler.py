"""컴파일러 테스트 — required_columns/financial_columns/position_rules 분리,
cross_above 경계, NaN→False, entry/exit 신호 손계산 대조 (명세 A5 §5·§6).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.compiler import (
    CompiledStrategy,
    PositionRules,
    compile_strategy,
    entry_signal,
    exit_signal,
)
from research_backtest.quant.strategy.indicators import compute_indicators
from research_backtest.quant.strategy.schema import StrategySpec, load_strategy_spec

DailyFrameFactory = Callable[..., pd.DataFrame]


def _spec(entry: dict[str, Any], exit_: dict[str, Any]) -> StrategySpec:
    """entry/exit만 바꿔가며 최소 StrategySpec을 만드는 헬퍼 (universe/execution은 고정값)."""
    return StrategySpec.model_validate(
        {
            "strategy_name": "TestStrategy",
            "universe": {"type": "single_asset", "tickers": ["000660"]},
            "entry": entry,
            "exit": exit_,
        }
    )


# --- README §23.4 기본 전략 컴파일 -------------------------------------------


def test_compile_readme_fixture_unmodified(earnings_flow_breakout_path: Path) -> None:
    """README §23.4 JSON을 그대로 로드·컴파일한다 (명세 A5 DoD 2)."""
    spec = load_strategy_spec(earnings_flow_breakout_path)
    compiled = compile_strategy(spec)

    assert compiled.required_columns == {
        "operating_income_yoy",
        "foreign_net_buy_20d",
        "close",
        "rolling_high_60_lag1",
        "sma_20",
    }
    assert compiled.financial_columns == {"operating_income_yoy"}
    assert compiled.position_rules == PositionRules(max_holding_days=60, stop_loss=-0.10)


def test_compile_rejects_unsupported_indicator() -> None:
    spec = _spec(
        entry={"all": [{"left": "sma_7", "operator": ">", "right": 1.0}]},
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    with pytest.raises(StrategyValidationError):
        compile_strategy(spec)


def test_compile_rejects_unsupported_indicator_referenced_as_right() -> None:
    entry = {"all": [{"left": "close", "operator": "cross_above", "right": "not_a_real_indicator"}]}
    spec = _spec(
        entry=entry,
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    with pytest.raises(StrategyValidationError):
        compile_strategy(spec)


# --- position_rules 분리 -----------------------------------------------------


def test_position_rules_separated_from_exit_condition_items() -> None:
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={
            "any": [
                {"left": "close", "operator": "<", "right": 0.0},
                {"type": "max_holding_days", "value": 30},
                {"type": "stop_loss", "value": -0.05},
            ]
        },
    )
    compiled = compile_strategy(spec)
    assert compiled.position_rules == PositionRules(max_holding_days=30, stop_loss=-0.05)


def test_position_rules_pure_rule_based_exit_signal_always_false(
    make_daily_frame: DailyFrameFactory,
) -> None:
    """exit.any가 규칙형 항목만 있으면 exit_signal은 항상 False다(엔진이 규칙을 직접 집행)."""
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={"any": [{"type": "max_holding_days", "value": 5}]},
    )
    compiled = compile_strategy(spec)
    assert compiled.position_rules.max_holding_days == 5
    assert compiled.position_rules.stop_loss is None

    daily = make_daily_frame(close=[float(i) for i in range(1, 11)])
    frame = compute_indicators(daily, compiled.required_columns)
    exit_series = exit_signal(compiled, frame)
    assert not exit_series.any()
    assert exit_series.name == "exit_signal"


def test_duplicate_max_holding_rule_rejected() -> None:
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={
            "any": [
                {"type": "max_holding_days", "value": 10},
                {"type": "max_holding_days", "value": 20},
            ]
        },
    )
    with pytest.raises(StrategyValidationError):
        compile_strategy(spec)


def test_duplicate_stop_loss_rule_rejected() -> None:
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={
            "any": [
                {"type": "stop_loss", "value": -0.05},
                {"type": "stop_loss", "value": -0.10},
            ]
        },
    )
    with pytest.raises(StrategyValidationError):
        compile_strategy(spec)


# --- entry/exit 손계산 대조 ---------------------------------------------------


def test_entry_signal_all_group_is_and(make_daily_frame: DailyFrameFactory) -> None:
    """all(AND) — 두 조건 모두 참인 행만 True (손계산 대조)."""
    close = [10.0, 20.0, 10.0, 20.0]
    volume = [100.0, 100.0, 200.0, 200.0]
    daily = make_daily_frame(close=close, volume=volume)
    spec = _spec(
        entry={
            "all": [
                {"left": "close", "operator": ">", "right": 15.0},
                {"left": "volume", "operator": ">", "right": 150.0},
            ]
        },
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)

    entry = entry_signal(compiled, frame)
    # row0: close=10(F) -> False; row1: close=20(T) but volume=100(F) -> False
    # row2: close=10(F) -> False; row3: close=20(T) and volume=200(T) -> True
    assert entry.tolist() == [False, False, False, True]
    assert entry.name == "entry_signal"


def test_entry_signal_any_group_is_or(make_daily_frame: DailyFrameFactory) -> None:
    """any(OR) — 하나만 참이어도 True (손계산 대조)."""
    close = [10.0, 20.0, 10.0, 20.0]
    volume = [100.0, 100.0, 200.0, 200.0]
    daily = make_daily_frame(close=close, volume=volume)
    spec = _spec(
        entry={
            "any": [
                {"left": "close", "operator": ">", "right": 15.0},
                {"left": "volume", "operator": ">", "right": 150.0},
            ]
        },
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)

    entry = entry_signal(compiled, frame)
    assert entry.tolist() == [False, True, True, True]


def test_cross_above_boundary_via_entry_signal(make_daily_frame: DailyFrameFactory) -> None:
    """cross_above 경계: 어제 이하 -> 오늘 초과 = True, 계속 위 = False (명세 A5 §6).

    right는 화이트리스트 지표여야 하므로(§21), 상수 취급을 위해 ``open``을
    일정 값으로 고정해 비교 대상("threshold" 역할)으로 쓴다.
    """
    close = [5.0, 5.0, 6.0, 7.0, 7.0, 3.0]
    open_ = [5.0] * 6
    daily = make_daily_frame(close=close, open_=open_)

    spec = _spec(
        entry={"all": [{"left": "close", "operator": "cross_above", "right": "open"}]},
        exit_={"any": [{"left": "close", "operator": "cross_below", "right": "open"}]},
    )
    compiled = compile_strategy(spec)
    assert compiled.required_columns == {"close", "open"}

    frame = compute_indicators(daily, compiled.required_columns)

    entry = entry_signal(compiled, frame)
    exit_ = exit_signal(compiled, frame)

    assert entry.tolist() == [False, False, True, False, False, False]
    assert exit_.tolist() == [False, False, False, False, False, True]


def test_between_operator_via_entry_signal(make_daily_frame: DailyFrameFactory) -> None:
    close = [1.0, 2.0, 3.0, 4.0, 5.0]
    daily = make_daily_frame(close=close)
    spec = _spec(
        entry={"all": [{"left": "close", "operator": "between", "right": [2.0, 4.0]}]},
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)
    entry = entry_signal(compiled, frame)
    assert entry.tolist() == [False, True, True, True, False]


def test_nan_in_required_column_is_false_not_error(make_daily_frame: DailyFrameFactory) -> None:
    """워밍업 NaN 구간(sma_20 등)에서 entry/exit는 에러 없이 False다."""
    close = [float(100 + i) for i in range(25)]
    daily = make_daily_frame(close=close)
    spec = _spec(
        entry={"all": [{"left": "close", "operator": "cross_above", "right": "sma_20"}]},
        exit_={"any": [{"left": "close", "operator": "cross_below", "right": "sma_20"}]},
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)

    assert frame["sma_20"].iloc[:19].isna().all()

    entry = entry_signal(compiled, frame)
    exit_ = exit_signal(compiled, frame)
    assert not entry.iloc[:19].any()
    assert not exit_.iloc[:19].any()
    assert not entry.isna().any()
    assert not exit_.isna().any()


def test_missing_required_column_raises_strategy_validation_error(
    make_daily_frame: DailyFrameFactory,
) -> None:
    daily = make_daily_frame(close=[1.0, 2.0, 3.0])
    spec = _spec(
        entry={"all": [{"left": "operating_income_yoy", "operator": ">", "right": 0.1}]},
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)  # operating_income_yoy 없음

    with pytest.raises(StrategyValidationError):
        entry_signal(compiled, frame)


def test_exit_signal_combines_multiple_condition_items_with_or(
    make_daily_frame: DailyFrameFactory,
) -> None:
    close = [10.0, 10.0, 10.0]
    daily = make_daily_frame(close=close, volume=[1.0, 500.0, 1.0])
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={
            "any": [
                {"left": "volume", "operator": ">", "right": 400.0},
                {"type": "max_holding_days", "value": 5},
            ]
        },
    )
    compiled = compile_strategy(spec)
    frame = compute_indicators(daily, compiled.required_columns)
    exit_ = exit_signal(compiled, frame)
    assert exit_.tolist() == [False, True, False]


# --- CompiledStrategy 자체 필드 계약 ------------------------------------------


def test_compiled_strategy_holds_original_spec() -> None:
    spec = _spec(
        entry={"all": [{"left": "close", "operator": ">", "right": 0.0}]},
        exit_={"any": [{"type": "max_holding_days", "value": 10}]},
    )
    compiled = compile_strategy(spec)
    assert isinstance(compiled, CompiledStrategy)
    assert compiled.spec is spec


# --- no-lookahead: entry/exit 신호도 절단 불변 --------------------------------


def test_entry_and_exit_signal_are_truncation_invariant() -> None:
    rng = np.random.default_rng(2026)
    n = 120
    dates = pd.bdate_range("2023-03-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1_000.0),
            "foreign_net_buy_value": rng.normal(0, 500, n),
            "institution_net_buy_value": rng.normal(0, 500, n),
        }
    ).set_index("date")

    spec = _spec(
        entry={
            "all": [
                {"left": "close", "operator": "cross_above", "right": "rolling_high_60_lag1"},
                {"left": "foreign_net_buy_20d", "operator": ">", "right": 0.0},
            ]
        },
        exit_={
            "any": [
                {"left": "close", "operator": "cross_below", "right": "sma_20"},
                {"type": "max_holding_days", "value": 60},
            ]
        },
    )
    compiled = compile_strategy(spec)

    full_frame = compute_indicators(daily, compiled.required_columns)
    full_entry = entry_signal(compiled, full_frame)
    full_exit = exit_signal(compiled, full_frame)

    cutoff = 80
    truncated_frame = compute_indicators(daily.iloc[:cutoff], compiled.required_columns)
    truncated_entry = entry_signal(compiled, truncated_frame)
    truncated_exit = exit_signal(compiled, truncated_frame)

    assert full_entry.iloc[:cutoff].tolist() == truncated_entry.tolist()
    assert full_exit.iloc[:cutoff].tolist() == truncated_exit.tolist()
