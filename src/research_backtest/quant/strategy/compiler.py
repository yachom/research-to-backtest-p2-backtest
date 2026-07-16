"""전략 컴파일러 — 검증 + 신호 계산기 컴파일 (README §22~§23, 명세 A5 §5).

:func:`compile_strategy`가 지표·연산자·구조 검증을 전부 끝내므로, 이후
:func:`entry_signal`/:func:`exit_signal` 호출 시점에는(프레임에 필요한
컬럼이 실제로 있는 한) 예외가 나지 않는다 — README M8 DoD("임의 Python
코드 실행 없음")를 지키며 화이트리스트 연산(비교·cross·between·AND·OR)만
해석한다.

조건 평가 규칙(명세 A5 §5):

- ``cross_above(l, r)`` = ``(l > r) & (l.shift(1) <= r.shift(1))`` — 오늘
  처음으로 상향 돌파(어제는 이하, 오늘은 초과)한 시점만 True.
- ``cross_below``는 대칭.
- ``between``은 ``[low, high]`` 폐구간(``low <= left <= high``).
- NaN이 관여하는 비교는 전부 False(신호 없음) — 워밍업 구간·수집 공백을
  "조건 미충족"으로 취급한다.
- ``ExitSpec.any``의 ``MaxHoldingRule``·``StopLossRule``은 포지션 상태
  (진입일·진입가)가 있어야 판정할 수 있으므로 여기서 신호로 계산하지
  않는다 — :class:`PositionRules`로 분리해 반환하고, 엔진(A6)이 보유
  포지션 상태를 기반으로 직접 집행한다. 나머지(Condition/ConditionGroup)
  항목만 ``exit_signal``로 컴파일한다.
"""

from __future__ import annotations

import operator as _op
from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.registry import IndicatorSource, resolve_indicator
from research_backtest.quant.strategy.schema import (
    Condition,
    ConditionGroup,
    ExitSpec,
    MaxHoldingRule,
    StopLossRule,
    StrategySpec,
)

ConditionNode = Condition | ConditionGroup

_COMPARATORS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    ">": _op.gt,
    ">=": _op.ge,
    "<": _op.lt,
    "<=": _op.le,
    "==": _op.eq,
}


class PositionRules(BaseModel):
    """엔진(A6)이 포지션 상태 기반으로 직접 집행하는 청산 규칙 (명세 A5 §5).

    조건 기반(exit_signal)과 달리 진입일·진입가 등 포지션 상태가 있어야
    판정할 수 있어 컴파일 단계에서 분리해 둔다.
    """

    max_holding_days: int | None = None  # 거래일 기준 (README §23.2)
    stop_loss: float | None = None  # 진입가 대비 손실률, 예: -0.10 (README §23.2)


class CompiledStrategy(BaseModel):
    """컴파일된 전략 — A6가 백테스트 실행에 사용하는 계약 (명세 A5 §5).

    - ``required_columns``: entry·exit(조건형)에 등장하는 전체 지표명(lag
      변형 포함, 프레임 컬럼명 그대로) — :func:`indicators.compute_indicators`의
      ``required`` 인자로 그대로 전달한다.
    - ``financial_columns``: ``required_columns`` 중 FINANCIAL 분류의
      **base**(lag 제거) 지표명 — A4 ``financial_metrics``의 실제 metric_id와
      1:1 대응하며, A6가 as-of join으로 공급해야 하는 컬럼 집합이다. lag가
      붙은 FINANCIAL 지표(예: ``operating_margin_lag1``)가 있다면 A6는 이
      base 컬럼을 먼저 join한 뒤 ``compute_indicators``를 호출해야 lag
      변형까지 채워진다(명세 A5 §4 compute_indicators 2단계).
    """

    # 명세 A5 §5의 클래스 골격을 그대로 따른다 — 현재 필드 구성(BaseModel·set[str]·
    # 중첩 BaseModel)만으로는 필요하지 않지만, 계약 파괴 없이 향후 필드가 늘어도
    # 안전하도록 유지한다.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: StrategySpec
    required_columns: set[str]
    financial_columns: set[str]
    position_rules: PositionRules


def compile_strategy(spec: StrategySpec) -> CompiledStrategy:
    """:class:`StrategySpec`을 검증하고 신호 계산 메타데이터를 컴파일한다 (명세 A5 §5).

    entry(all 그룹)·exit.any(조건형 항목)에 등장하는 모든 지표명을
    :func:`registry.resolve_indicator`로 검증한다 — 미지원 지표는 여기서
    :class:`StrategyValidationError`로 거부된다(§21 화이트리스트 밖 이름은
    스키마 검증만으로는 걸러지지 않으므로, 컴파일이 곧 지표 검증 시점이다).
    """
    exit_condition_items = _exit_condition_items(spec.exit)

    all_names: set[str] = _collect_names(spec.entry)
    for item in exit_condition_items:
        all_names |= _collect_names(item)

    resolved = {name: resolve_indicator(name) for name in all_names}

    required_columns = set(all_names)
    financial_columns = {
        info.base for info in resolved.values() if info.source is IndicatorSource.FINANCIAL
    }

    return CompiledStrategy(
        spec=spec,
        required_columns=required_columns,
        financial_columns=financial_columns,
        position_rules=_build_position_rules(spec.exit),
    )


def entry_signal(compiled: CompiledStrategy, frame: pd.DataFrame) -> pd.Series:
    """진입 신호 — ``spec.entry``(all 그룹)를 평가한 bool Series(index=frame.index).

    ``frame``은 ``compiled.required_columns``를 전부 포함해야 한다
    (:func:`indicators.compute_indicators` 실행 + A6의 재무 as-of join
    이후). 없으면 :class:`StrategyValidationError`.
    """
    _check_required_columns(compiled, frame)
    return _eval_node(compiled.spec.entry, frame).rename("entry_signal")


def exit_signal(compiled: CompiledStrategy, frame: pd.DataFrame) -> pd.Series:
    """조건 기반 청산 신호 — ``exit.any`` 중 Condition/ConditionGroup만 평가한다.

    ``frame`` 컬럼 요구사항은 :func:`entry_signal`과 동일하다. 규칙 기반
    항목(MaxHoldingRule·StopLossRule)은 ``compiled.position_rules``로 이미
    분리되어 있으며 엔진(A6)이 포지션 상태로 별도 집행한다 — 조건형 항목이
    하나도 없으면 항상 False인 Series를 반환한다(순수 규칙 기반 청산 전략).
    """
    _check_required_columns(compiled, frame)
    items = _exit_condition_items(compiled.spec.exit)
    if not items:
        return pd.Series(False, index=frame.index, name="exit_signal")
    combined = pd.Series(False, index=frame.index)
    for item in items:
        combined = combined | _eval_node(item, frame)
    return combined.fillna(False).rename("exit_signal")


# --- 내부 구현 ---------------------------------------------------------------


def _collect_names(node: ConditionNode) -> set[str]:
    """조건 트리(all/any 재귀)에서 등장하는 모든 지표명(left + 컬럼 참조 right)을 모은다."""
    if isinstance(node, Condition):
        condition_names = {node.left}
        if isinstance(node.right, str):
            condition_names.add(node.right)
        return condition_names
    items = node.all if node.all is not None else node.any
    assert items is not None  # ConditionGroup 스키마가 all/any 중 정확히 하나를 보장한다
    group_names: set[str] = set()
    for item in items:
        group_names |= _collect_names(item)
    return group_names


def _exit_condition_items(exit_spec: ExitSpec) -> list[ConditionNode]:
    """``exit.any``에서 조건형(Condition/ConditionGroup) 항목만 추린다."""
    return [item for item in exit_spec.any if isinstance(item, Condition | ConditionGroup)]


def _build_position_rules(exit_spec: ExitSpec) -> PositionRules:
    """``exit.any``의 규칙형(MaxHoldingRule·StopLossRule) 항목을 분리한다.

    같은 규칙이 두 번 이상 지정되면 어느 값을 쓸지 모호하므로 명시적으로 거부한다.
    """
    max_holding_days: int | None = None
    stop_loss: float | None = None
    for item in exit_spec.any:
        if isinstance(item, MaxHoldingRule):
            if max_holding_days is not None:
                raise StrategyValidationError(
                    "exit.any에 max_holding_days 규칙이 중복 지정되었습니다."
                )
            max_holding_days = item.value
        elif isinstance(item, StopLossRule):
            if stop_loss is not None:
                raise StrategyValidationError("exit.any에 stop_loss 규칙이 중복 지정되었습니다.")
            stop_loss = item.value
    return PositionRules(max_holding_days=max_holding_days, stop_loss=stop_loss)


def _check_required_columns(compiled: CompiledStrategy, frame: pd.DataFrame) -> None:
    missing = sorted(compiled.required_columns - set(frame.columns))
    if missing:
        raise StrategyValidationError(
            f"프레임에 필요한 지표 컬럼이 없습니다: {missing}. "
            "indicators.compute_indicators()와 A6의 재무 as-of join을 먼저 수행하세요."
        )


def _eval_node(node: ConditionNode, frame: pd.DataFrame) -> pd.Series:
    if isinstance(node, ConditionGroup):
        return _eval_group(node, frame)
    return _eval_condition(node, frame)


def _eval_group(group: ConditionGroup, frame: pd.DataFrame) -> pd.Series:
    if group.all is not None:
        combined = pd.Series(True, index=frame.index)
        for item in group.all:
            combined = combined & _eval_node(item, frame)
        return combined.fillna(False)
    assert group.any is not None
    combined = pd.Series(False, index=frame.index)
    for item in group.any:
        combined = combined | _eval_node(item, frame)
    return combined.fillna(False)


def _eval_condition(cond: Condition, frame: pd.DataFrame) -> pd.Series:
    left: pd.Series = frame[cond.left]
    if cond.operator == "between":
        assert isinstance(cond.right, list)  # schema가 between의 right를 [low, high]로 보장
        low, high = cond.right
        result = (left >= low) & (left <= high)
    elif cond.operator == "cross_above":
        right = _right_operand(cond.right, frame)
        result = (left > right) & (left.shift(1) <= right.shift(1))
    elif cond.operator == "cross_below":
        right = _right_operand(cond.right, frame)
        result = (left < right) & (left.shift(1) >= right.shift(1))
    else:
        right = _right_operand(cond.right, frame)
        result = _COMPARATORS[cond.operator](left, right)
    return result.fillna(False)


def _right_operand(right: float | int | str | list[float], frame: pd.DataFrame) -> pd.Series:
    """between을 제외한 연산자의 right를 frame.index 기준 Series로 만든다.

    문자열이면 컬럼 참조, 숫자면 상수를 프레임 전체에 브로드캐스트한다 —
    ``cross_above``/``cross_below``가 ``.shift(1)``을 균일하게 쓸 수 있도록
    항상 Series를 반환한다(스칼라 상수와 비교하는 ``>``/``>=`` 등에도 그대로 쓸 수 있다).
    """
    if isinstance(right, str):
        column: pd.Series = frame[right]
        return column
    assert isinstance(right, int | float)
    return pd.Series(float(right), index=frame.index)
