"""전략 DSL 초안 생성기 (명세 docs/specs/W3b-candidates-strategy.md §3.1).

승인된 사용자 투자 가설(:class:`HumanInvestmentHypothesis`)을 LLM으로 전략
DSL(README §20~§23) JSON **초안**으로 번역한다. 이 모듈이 만드는 것은 사용자가
검토·수정한 뒤 승인해야 하는 초안일 뿐이다(docs/AI_ROLE_BOUNDARY.md — AI는
최종 투자 의견을 확정하지 않는다). 최종 승인은
``app.commands.hitl_flow.approve_strategy``\\ (``approve-strategy``)가 사람의
:class:`~research_backtest.core.hitl.models.StrategyReview`\\ 로 수행한다.

게이트(방어선 중첩 — :mod:`research_backtest.quant.backtest.runner`\\ 의
``execute_approved_strategy`` 패턴과 동일): :func:`draft_strategy`\\ 는
**함수 첫 줄에서**
:func:`~research_backtest.core.hitl.gates.ensure_hypothesis_approved`\\ 를
호출해 미승인 가설의 전략 변환을 즉시 차단한다 — 호출부(``generate-strategy-
draft`` CLI)가 이미 같은 게이트를 검사했더라도, 이 함수가 다른 경로로
호출되는 경우(테스트·향후 배치 등)까지 대비한 두 번째 방어선이다.

LLM 응답 검증(재시도 루프에 주입되는 validator, 명세 §3.1) 순서: JSON dict →
:func:`~research_backtest.quant.strategy.schema.parse_strategy_spec` →
:func:`~research_backtest.quant.strategy.compiler.compile_strategy`\\ (지표
화이트리스트·청산 규칙 중복까지 검증) → ``universe.tickers == [stock_code]`` →
``strategy_name`` 비어있지 않음 → ``execution``\\ 이 signal_time="close"·
trade_time="next_open" 고정(README §22 룩어헤드 방지). 위반은 전부
:class:`~research_backtest.core.exceptions.StrategyValidationError`\\ 로 던져
:func:`~research_backtest.core.llm.json_call.complete_validated`\\ 가 오류
요지를 덧붙인 재요청으로 재시도하게 한다. tickers·strategy_name 위반은
스키마만으로는 걸러지지 않아 실제로 재시도를 유발할 수 있다 — execution
위반은 ``ExecutionSpec``\\ 의 ``Literal`` 타입이 이미 파싱 단계에서 걸러내지만
(따라서 사실상 도달하지 않는 방어선이지만), 명세가 명시적으로 요구하는
이중 방어선이며 스키마가 느슨해지더라도 안전하도록 유지한다.

성공한 초안은 원문 LLM 텍스트가 아니라
:meth:`StrategySpec.model_dump`\\ (``exclude_none=True`` — ``ConditionGroup``\\ 의
미사용 분기(all/any/not)가 ``null``로 섞여 나오지 않도록 함) 결과를 반환한다.
사용자가 ``strategy_draft.json``\\ 만 보고도 실제 적용될 모든 필드값(생략된
기본값 포함)을 확인할 수 있게 하기 위해서다.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.core.hitl import gates
from research_backtest.core.hitl.models import HumanInvestmentHypothesis
from research_backtest.core.llm.client import LlmCallMetadata, LlmTextClient
from research_backtest.core.llm.json_call import complete_validated
from research_backtest.core.llm.prompts import load_prompt
from research_backtest.quant.strategy.compiler import compile_strategy
from research_backtest.quant.strategy.registry import (
    FINANCIAL_INDICATORS,
    FLOW_INDICATORS,
    PRICE_INDICATORS,
)
from research_backtest.quant.strategy.schema import parse_strategy_spec

#: ``quant/prompts/`` 디렉토리 — 소스 파일 위치 기준(실행 CWD 무관, 명세 §3.1).
#: ``configs/*.yaml``(레포 루트 상대 경로)과 달리, 프롬프트는 패키지 소스에
#: 함께 배포되는 자산이라 ``__file__`` 기준으로 위치를 잡는다(tests/unit/
#: strategy/conftest.py의 ``FIXTURE_DIR`` 계산과 같은 방식).
DEFAULT_PROMPTS_DIR: Path = Path(__file__).resolve().parent.parent / "prompts"

_PROMPT_NAME = "strategy_translation"
_PROMPT_VERSION = 1

_SYSTEM_PROMPT = (
    "당신은 사용자가 이미 승인한 투자 가설을 전략 DSL JSON 초안으로 옮기는 "
    "보조자다. 전략의 최종 확정 권한은 사용자에게 있으며, 당신의 결과물은 "
    "검토·수정을 전제로 한 초안일 뿐이다. 설명이나 코드펜스 없이 유효한 "
    "JSON 객체 하나만 출력하라."
)

# README §23.4 전략 JSON과 내용이 동일하다(=tests/fixtures/strategy/
# earnings_flow_breakout.json — tests/unit/strategy/test_draft.py가 두 내용의
# 일치를 회귀 테스트로 고정한다). 프로덕션 코드가 tests/ 경로를 참조하지
# 않도록 이 상수로 값을 복제해 둔다. 프롬프트에는 json.dumps로 렌더링한
# 문자열을 넣는다 — 파이썬 딕셔너리 리터럴 자체는 str.format 이스케이프
# 문제와 무관하다(PromptTemplate.render는 이 상수의 dump 결과를 kwarg 값으로만
# 받으므로 중괄호 이스케이스가 필요 없다).
_EXAMPLE_STRATEGY_SPEC: dict[str, object] = {
    "strategy_name": "EarningsFlowBreakout",
    "version": "1.0",
    "universe": {"type": "single_asset", "tickers": ["000660"]},
    "entry": {
        "all": [
            {"left": "operating_income_yoy", "operator": ">", "right": 0.20},
            {"left": "foreign_net_buy_20d", "operator": ">", "right": 0},
            {
                "left": "close",
                "operator": "cross_above",
                "right": "rolling_high_60_lag1",
            },
        ]
    },
    "exit": {
        "any": [
            {"left": "close", "operator": "cross_below", "right": "sma_20"},
            {"type": "max_holding_days", "value": 60},
            {"type": "stop_loss", "value": -0.10},
        ]
    },
    "execution": {"signal_time": "close", "trade_time": "next_open"},
}


def draft_strategy(
    hypothesis: HumanInvestmentHypothesis,
    *,
    stock_code: str,
    client: LlmTextClient,
    prompts_dir: Path,
    max_attempts: int,
) -> tuple[dict[str, object], LlmCallMetadata]:
    """승인된 투자 가설을 전략 DSL 초안(dict)으로 변환한다 (명세 §3.1).

    반환하는 dict는 :func:`~research_backtest.quant.strategy.schema.
    parse_strategy_spec`\\ 과 :func:`~research_backtest.quant.strategy.compiler.
    compile_strategy`\\ 를 이미 통과한 것이 보장된다 — 호출부
    (``generate-strategy-draft``)는 추가 검증 없이 바로
    ``store.save_strategy_draft``\\ 에 저장할 수 있다.
    """
    gates.ensure_hypothesis_approved(hypothesis)

    prompt = load_prompt(prompts_dir, _PROMPT_NAME, _PROMPT_VERSION)
    user_prompt = prompt.render(
        stock_code=stock_code,
        thesis=hypothesis.thesis,
        economic_rationale=hypothesis.economic_rationale,
        expected_mechanism=hypothesis.expected_mechanism,
        selected_variables=", ".join(hypothesis.selected_variables),
        expected_direction=hypothesis.expected_direction,
        investment_horizon_days=str(hypothesis.investment_horizon_days),
        financial_indicators=_bulleted(FINANCIAL_INDICATORS),
        price_indicators=_bulleted(PRICE_INDICATORS),
        flow_indicators=_bulleted(FLOW_INDICATORS),
        example_strategy_json=json.dumps(_EXAMPLE_STRATEGY_SPEC, ensure_ascii=False, indent=2),
    )

    def _validate(payload: object) -> dict[str, object]:
        return _validate_draft(payload, stock_code=stock_code)

    return complete_validated(
        client,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        validator=_validate,
        max_attempts=max_attempts,
    )


def _bulleted(names: frozenset[str]) -> str:
    """지표 화이트리스트를 결정적 순서(알파벳순)의 불릿 목록 문자열로 렌더링한다."""
    return "\n".join(f"- {name}" for name in sorted(names))


def _validate_draft(payload: object, *, stock_code: str) -> dict[str, object]:
    """전략 초안 validator (명세 §3.1) — parse → compile → tickers → strategy_name
    → execution 순으로 검증한다.

    전부 통과하면 :meth:`StrategySpec.model_dump`\\ (``exclude_none=True``)
    결과를 반환한다. 위반은 전부
    :class:`~research_backtest.core.exceptions.StrategyValidationError`\\ 로
    던져 :func:`~research_backtest.core.llm.json_call.complete_validated`\\ 의
    재시도 피드백이 되게 한다.
    """
    if not isinstance(payload, dict):
        raise StrategyValidationError(
            f"전략 초안은 JSON 객체(dict)여야 합니다 — 받은 타입: {type(payload).__name__}"
        )

    spec = parse_strategy_spec(payload)
    compile_strategy(spec)  # 지표 화이트리스트·청산 규칙 중복 등(§21) 검증

    if spec.universe.tickers != [stock_code]:
        raise StrategyValidationError(
            f"universe.tickers는 정확히 [{stock_code!r}]이어야 합니다 — "
            f"받은 값: {spec.universe.tickers!r}"
        )
    if not spec.strategy_name.strip():
        raise StrategyValidationError("strategy_name은 비어 있을 수 없습니다.")
    if spec.execution.signal_time != "close" or spec.execution.trade_time != "next_open":
        raise StrategyValidationError(
            "execution은 signal_time='close'·trade_time='next_open'로 고정해야 합니다"
            f"(README §22 룩어헤드 방지) — 받은 값: signal_time={spec.execution.signal_time!r}, "
            f"trade_time={spec.execution.trade_time!r}"
        )

    return spec.model_dump(mode="json", exclude_none=True)


__all__ = ["DEFAULT_PROMPTS_DIR", "draft_strategy"]
