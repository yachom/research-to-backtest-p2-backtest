"""전략 JSON pydantic 스키마 테스트 (README §23.4, 명세 A5 §2·§6).

핵심: README §23.4 JSON이 **무수정**으로 검증을 통과해야 하고(DoD 2),
알 수 없는 필드·중복 all/any·미지원 연산자·미지원 ``not``은 전부 명시적
:class:`StrategyValidationError`로 거부되어야 한다(DoD 4).
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.schema import (
    Condition,
    ConditionGroup,
    ExecutionSpec,
    MaxHoldingRule,
    StopLossRule,
    StrategySpec,
    load_strategy_spec,
    parse_strategy_spec,
)


def test_readme_fixture_validates_unmodified(earnings_flow_breakout_raw: dict[str, Any]) -> None:
    """README §23.4 JSON이 그대로 StrategySpec 검증을 통과한다 (명세 A5 DoD 2)."""
    spec = parse_strategy_spec(earnings_flow_breakout_raw)

    assert spec.strategy_name == "EarningsFlowBreakout"
    assert spec.version == "1.0"
    assert spec.universe.type == "single_asset"
    assert spec.universe.tickers == ["000660"]
    assert spec.entry.all is not None
    assert len(spec.entry.all) == 3
    assert len(spec.exit.any) == 3  # ExitSpec.any는 항상 리스트(Optional 아님)
    assert spec.execution.signal_time == "close"
    assert spec.execution.trade_time == "next_open"


def test_load_strategy_spec_from_fixture_file(earnings_flow_breakout_path: Path) -> None:
    """load_strategy_spec()으로 fixture 파일을 직접 로드해도 동일하게 통과한다 (A6·C2 재사용)."""
    spec = load_strategy_spec(earnings_flow_breakout_path)
    assert spec.strategy_name == "EarningsFlowBreakout"


def test_load_strategy_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(StrategyValidationError):
        load_strategy_spec(tmp_path / "does_not_exist.json")


def test_load_strategy_spec_malformed_json_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StrategyValidationError):
        load_strategy_spec(bad_file)


def test_rejects_unknown_top_level_field(earnings_flow_breakout_raw: dict[str, Any]) -> None:
    earnings_flow_breakout_raw["unknown_field"] = 1
    with pytest.raises(StrategyValidationError):
        parse_strategy_spec(earnings_flow_breakout_raw)


def test_rejects_unknown_condition_field(earnings_flow_breakout_raw: dict[str, Any]) -> None:
    earnings_flow_breakout_raw["entry"]["all"][0]["extra"] = "nope"
    with pytest.raises(StrategyValidationError):
        parse_strategy_spec(earnings_flow_breakout_raw)


def test_rejects_all_and_any_together(earnings_flow_breakout_raw: dict[str, Any]) -> None:
    condition = earnings_flow_breakout_raw["entry"]["all"][0]
    earnings_flow_breakout_raw["entry"]["any"] = [condition]
    with pytest.raises(StrategyValidationError):
        parse_strategy_spec(earnings_flow_breakout_raw)


def test_condition_group_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="all/any/not"):
        ConditionGroup.model_validate({})


def test_condition_group_rejects_empty_all_list() -> None:
    with pytest.raises(ValueError, match="빈 리스트"):
        ConditionGroup.model_validate({"all": []})


def test_rejects_unsupported_comparison_operator(
    earnings_flow_breakout_raw: dict[str, Any],
) -> None:
    earnings_flow_breakout_raw["entry"]["all"][0]["operator"] = "!="
    with pytest.raises(StrategyValidationError):
        parse_strategy_spec(earnings_flow_breakout_raw)


def test_not_operator_is_recognized_but_explicitly_unsupported() -> None:
    """'not'은 조용히 무시되지 않는다 — extra=forbid의 일반 오류가 아니라 전용 메시지여야 한다."""
    with pytest.raises(ValueError, match="not") as excinfo:
        ConditionGroup.model_validate({"not": {"left": "close", "operator": ">", "right": 1.0}})
    assert "미지원" in str(excinfo.value)


def test_between_requires_two_element_list() -> None:
    with pytest.raises(ValueError, match="between"):
        Condition.model_validate({"left": "rsi_14", "operator": "between", "right": [30.0]})


def test_between_rejects_non_list_right() -> None:
    with pytest.raises(ValueError, match="between"):
        Condition.model_validate({"left": "rsi_14", "operator": "between", "right": 30.0})


def test_between_requires_low_le_high() -> None:
    with pytest.raises(ValueError, match="low"):
        Condition.model_validate({"left": "rsi_14", "operator": "between", "right": [70.0, 30.0]})


def test_between_accepts_valid_bounds() -> None:
    cond = Condition.model_validate(
        {"left": "rsi_14", "operator": "between", "right": [30.0, 70.0]}
    )
    assert cond.right == [30.0, 70.0]


def test_non_between_operator_rejects_list_right() -> None:
    with pytest.raises(ValueError, match="between 전용"):
        Condition.model_validate({"left": "close", "operator": ">", "right": [1.0, 2.0]})


def test_universe_rejects_empty_tickers() -> None:
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(
            {
                "strategy_name": "X",
                "universe": {"type": "single_asset", "tickers": []},
                "entry": {"all": [{"left": "close", "operator": ">", "right": 1}]},
                "exit": {"any": [{"type": "max_holding_days", "value": 10}]},
            }
        )


def test_universe_rejects_multiple_tickers() -> None:
    with pytest.raises(ValidationError):
        StrategySpec.model_validate(
            {
                "strategy_name": "X",
                "universe": {"type": "single_asset", "tickers": ["000660", "005930"]},
                "entry": {"all": [{"left": "close", "operator": ">", "right": 1}]},
                "exit": {"any": [{"type": "max_holding_days", "value": 10}]},
            }
        )


def test_execution_defaults_when_omitted() -> None:
    spec = StrategySpec.model_validate(
        {
            "strategy_name": "X",
            "universe": {"type": "single_asset", "tickers": ["000660"]},
            "entry": {"all": [{"left": "close", "operator": ">", "right": 1}]},
            "exit": {"any": [{"type": "max_holding_days", "value": 10}]},
        }
    )
    assert spec.execution == ExecutionSpec(signal_time="close", trade_time="next_open")


def test_max_holding_rule_requires_positive_value() -> None:
    with pytest.raises(ValidationError):
        MaxHoldingRule.model_validate({"type": "max_holding_days", "value": 0})


def test_stop_loss_rule_requires_negative_value() -> None:
    with pytest.raises(ValidationError):
        StopLossRule.model_validate({"type": "stop_loss", "value": 0.0})


def test_stop_loss_rule_rejects_positive_value() -> None:
    with pytest.raises(ValidationError):
        StopLossRule.model_validate({"type": "stop_loss", "value": 0.10})


def test_exit_spec_disambiguates_condition_and_rules(
    earnings_flow_breakout_raw: dict[str, Any],
) -> None:
    """ExitSpec.any의 Union(Condition|ConditionGroup|MaxHoldingRule|StopLossRule)이
    항목별로 올바른 타입으로 해석되는지 확인한다."""
    spec = parse_strategy_spec(earnings_flow_breakout_raw)
    kinds = [type(item).__name__ for item in spec.exit.any]
    assert kinds == ["Condition", "MaxHoldingRule", "StopLossRule"]
