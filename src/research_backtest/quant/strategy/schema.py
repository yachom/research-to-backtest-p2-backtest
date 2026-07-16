"""전략 DSL — 전략 JSON pydantic 스키마 (README §20~§23, 명세 A5 §2).

이 모듈은 전략 JSON의 **구조**만 검증한다 — 지표명이 실제로 허용 목록에
있는지(README §21)는 다루지 않는다(그건 :mod:`registry`·:mod:`compiler`의
책임, 명세 A5 §1 모듈 배치). 여기서는:

- README §23.4의 JSON이 어떤 수정도 없이 그대로 검증을 통과해야 한다.
- 알 수 없는 필드는 거부한다(``extra="forbid"``).
- 논리 연산자는 구조로 표현한다 — ``all``=AND, ``any``=OR. ``not``(README
  §21.4)은 자리만 두고(``ConditionGroup.not``), 사용되면 "조용히 무시"하지
  않고 명시적으로 미지원 오류를 낸다(명세 §2 마지막 불릿).

임의 Python 코드 실행은 없다 — 여기서 만들어지는 것은 선언적 데이터
구조뿐이며, 해석은 :mod:`compiler`가 화이트리스트 연산만으로 수행한다
(README M8 DoD).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from research_backtest.core.exceptions import StrategyValidationError

ComparisonOperator = Literal[">", ">=", "<", "<=", "==", "cross_above", "cross_below", "between"]


class _StrictModel(BaseModel):
    """전략 JSON 구성 모델의 공통 베이스 — 알 수 없는 필드는 항상 거부한다."""

    model_config = ConfigDict(extra="forbid")


class Condition(_StrictModel):
    """단일 비교 조건 (README §21.4 비교 연산자, 명세 A5 §2).

    ``left``는 지표명(레지스트리 검증은 compiler 단계), ``right``는 숫자
    상수·지표명(컬럼 참조)·``between``용 ``[low, high]`` 중 하나다.
    """

    left: str
    operator: ComparisonOperator
    right: float | int | str | list[float]

    @model_validator(mode="after")
    def _check_right_shape(self) -> Condition:
        if self.operator == "between":
            if not isinstance(self.right, list):
                raise ValueError(
                    f"between 연산자는 right가 [low, high] 리스트여야 합니다: {self.right!r}"
                )
            if len(self.right) != 2:
                raise ValueError(
                    f"between 연산자의 right는 정확히 2개(low, high)여야 합니다: {self.right!r}"
                )
            low, high = self.right
            if low > high:
                raise ValueError(f"between 연산자는 low<=high여야 합니다: {self.right!r}")
        elif isinstance(self.right, list):
            raise ValueError(
                f"연산자 {self.operator!r}에는 리스트 right를 사용할 수 없습니다(between 전용)."
            )
        return self


class ConditionGroup(_StrictModel):
    """진입/청산 조건의 재귀 그룹 — ``all``(AND)·``any``(OR) 중 정확히 하나
    (README §21.4, 명세 A5 §2).

    ``not``은 README §21.4의 논리 연산자 목록에는 있으나 명세 A5 §2가 준
    스키마 골격에는 구조가 없다 — 여기서는 자리만 만들어 두고(``not``
    alias), 실제로 사용되면 "미지원"임을 명시적으로 알린다(조용한 무시
    금지, 명세 §2 마지막 불릿).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    all: list[Condition | ConditionGroup] | None = None
    any: list[Condition | ConditionGroup] | None = None
    not_: Condition | ConditionGroup | None = Field(default=None, alias="not")

    @model_validator(mode="after")
    def _check_exactly_one_branch(self) -> ConditionGroup:
        branches = {"all": self.all, "any": self.any, "not": self.not_}
        provided = [key for key, value in branches.items() if value is not None]
        if len(provided) != 1:
            raise ValueError(
                "ConditionGroup은 all/any/not 중 정확히 하나만 지정해야 합니다 "
                f"(지정됨: {provided or '없음'})."
            )
        if provided[0] == "not":
            raise ValueError(
                "연산자 'not'은 README §21.4에 정의되어 있으나 MVP 스키마(명세 A5 §2)에서는 "
                "미지원입니다. all/any 조합으로 조건을 다시 작성하세요."
            )
        if provided[0] == "all" and not self.all:
            raise ValueError("all은 빈 리스트일 수 없습니다.")
        if provided[0] == "any" and not self.any:
            raise ValueError("any는 빈 리스트일 수 없습니다.")
        return self


ConditionGroup.model_rebuild()


class MaxHoldingRule(_StrictModel):
    """최대 보유기간 청산 규칙 — 거래일 기준 (README §23.2, 명세 A5 §2)."""

    type: Literal["max_holding_days"]
    value: int = Field(gt=0)


class StopLossRule(_StrictModel):
    """손절 청산 규칙 — 진입가 대비 손실률 (README §23.2, 명세 A5 §2).

    ``value``는 예: ``-0.10``(진입가 대비 -10%). 손실률이므로 음수만 허용한다.
    """

    type: Literal["stop_loss"]
    value: float = Field(lt=0)


ExitRule = Condition | ConditionGroup | MaxHoldingRule | StopLossRule


class ExitSpec(_StrictModel):
    """청산 조건 — 조건 기반(Condition/ConditionGroup)과 규칙 기반
    (MaxHoldingRule/StopLossRule)을 ``any``(OR) 하나로 섞어 표현한다
    (README §23.2, 명세 A5 §2).

    규칙 기반 항목은 :func:`compiler.compile_strategy`가
    ``PositionRules``로 분리하고, 나머지 조건 항목만 ``exit_signal``로
    컴파일한다(명세 A5 §5).
    """

    any: list[ExitRule] = Field(min_length=1)


class ExecutionSpec(_StrictModel):
    """체결 규칙 — README §23.3, MVP는 값이 고정이다."""

    signal_time: Literal["close"] = "close"
    trade_time: Literal["next_open"] = "next_open"


class UniverseSpec(_StrictModel):
    """전략 대상 유니버스 (README §23.4). MVP는 단일 종목만 지원한다."""

    type: Literal["single_asset"]
    tickers: list[str] = Field(min_length=1, max_length=1)


class StrategySpec(_StrictModel):
    """전략 JSON 최상위 스키마 — README §23.4가 그대로 통과해야 한다."""

    strategy_name: str
    version: str = "1.0"
    universe: UniverseSpec
    entry: ConditionGroup
    exit: ExitSpec
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)


def parse_strategy_spec(data: dict[str, Any] | str) -> StrategySpec:
    """전략 JSON(dict 또는 JSON 문자열)을 검증해 :class:`StrategySpec`으로 변환한다.

    구조·타입 위반(알 수 없는 필드, all/any 동시 지정, 잘못된 연산자,
    ``not`` 사용 등)은 모두 :class:`StrategyValidationError`로 통일한다
    (명세 A5 DoD 4) — 호출자는 pydantic 예외 타입을 알 필요가 없다.
    """
    try:
        if isinstance(data, str):
            return StrategySpec.model_validate_json(data)
        return StrategySpec.model_validate(data)
    except ValidationError as exc:
        raise StrategyValidationError(f"전략 JSON 검증 실패: {exc}") from exc


def load_strategy_spec(path: Path) -> StrategySpec:
    """전략 JSON 파일을 읽어 검증한다 — A6·C2가 fixture/사용자 전략을 로드할 때 사용

    (명세 A5 §5, ``tests/fixtures/strategy/earnings_flow_breakout.json`` 참고).
    파일이 없거나 JSON 파싱에 실패해도 :class:`StrategyValidationError`로 통일한다.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrategyValidationError(f"전략 JSON 파일을 읽을 수 없습니다: {path}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StrategyValidationError(f"전략 JSON 파싱 실패({path}): {exc}") from exc
    return parse_strategy_spec(raw)
