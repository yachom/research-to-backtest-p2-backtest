"""quant/strategy/draft.py 단위 테스트 (명세 docs/specs/W3b-candidates-strategy.md §3.1, §4).

전부 오프라인 — LLM 호출은 :class:`FakeLlmClient`\\ 로 대체한다. 검증 대상:
게이트(미승인 가설은 LLM 호출조차 하지 않는다), validator 재시도 경로
(컴파일 실패·tickers 불일치·strategy_name 공백 → 피드백 재시도 → 성공),
재시도 소진, 반환 dict의 정규화(exclude_none으로 ConditionGroup의 미사용
분기가 null로 섞이지 않음), 프롬프트에 가설 필드·지표 화이트리스트가 실제로
렌더링되는지.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from research_backtest.core.exceptions import ApprovalGateError, DataValidationError
from research_backtest.core.hitl.models import HumanInvestmentHypothesis
from research_backtest.core.llm.testing import FakeLlmClient
from research_backtest.quant.strategy.compiler import compile_strategy
from research_backtest.quant.strategy.draft import (
    _EXAMPLE_STRATEGY_SPEC,
    DEFAULT_PROMPTS_DIR,
    draft_strategy,
)
from research_backtest.quant.strategy.schema import parse_strategy_spec

STOCK_CODE = "000660"


def _hypothesis(**overrides: Any) -> HumanInvestmentHypothesis:
    payload: dict[str, Any] = {
        "hypothesis_id": "HYP-1",
        "view_id": "VIEW-1",
        "author": "홍길동",
        "thesis": "HBM 비중 확대가 이익률을 컨센서스 이상으로 끌어올린다.",
        "economic_rationale": "HBM 마진이 legacy 대비 높다.",
        "expected_mechanism": "ASP 상승 → 이익률 개선",
        "selected_variables": ["operating_income_yoy"],
        "expected_direction": "up",
        "investment_horizon_days": 90,
        "evidence_ids": ["EVID-001"],
        "falsification_conditions": ["2개 분기 연속 컨센서스 하회 시 기각"],
        "limitations": [],
        "status": "DRAFT",
        "created_at": "2026-07-14T11:00:00+09:00",
        "updated_at": "2026-07-14T11:00:00+09:00",
    }
    payload.update(overrides)
    return HumanInvestmentHypothesis.model_validate(payload)


def _approved_hypothesis(**overrides: Any) -> HumanInvestmentHypothesis:
    payload: dict[str, Any] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-07-14T12:00:00+09:00",
    }
    payload.update(overrides)
    return _hypothesis(**payload)


def _valid_draft_json(*, stock_code: str = STOCK_CODE, strategy_name: str = "TestDraft") -> str:
    payload = {
        "strategy_name": strategy_name,
        "version": "1.0",
        "universe": {"type": "single_asset", "tickers": [stock_code]},
        "entry": {"all": [{"left": "operating_income_yoy", "operator": ">", "right": 0.2}]},
        "exit": {"any": [{"type": "max_holding_days", "value": 60}]},
        "execution": {"signal_time": "close", "trade_time": "next_open"},
    }
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 게이트 — 함수 첫 줄
# ---------------------------------------------------------------------------


def test_draft_strategy_gate_blocks_unapproved_hypothesis() -> None:
    hypothesis = _hypothesis(status="DRAFT")
    client = FakeLlmClient([_valid_draft_json()])

    with pytest.raises(ApprovalGateError):
        draft_strategy(
            hypothesis,
            stock_code=STOCK_CODE,
            client=client,
            prompts_dir=DEFAULT_PROMPTS_DIR,
            max_attempts=3,
        )
    assert client.calls == []  # 게이트가 LLM 호출 이전에 막는다


# ---------------------------------------------------------------------------
# 성공 경로 — 반환 dict의 형태
# ---------------------------------------------------------------------------


def test_draft_strategy_succeeds_first_attempt_and_normalizes_dict() -> None:
    hypothesis = _approved_hypothesis()
    client = FakeLlmClient([_valid_draft_json()])

    draft, metadata = draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=3,
    )

    assert metadata.num_attempts == 1
    assert draft["universe"] == {"type": "single_asset", "tickers": [STOCK_CODE]}
    assert draft["execution"] == {"signal_time": "close", "trade_time": "next_open"}
    # exclude_none=True 정규화 — ConditionGroup의 미사용 분기(any/not)가 섞여 나오지 않는다.
    entry = draft["entry"]
    assert isinstance(entry, dict)
    assert "all" in entry
    assert "any" not in entry
    assert "not" not in entry
    assert "not_" not in entry
    # 반환값은 parse+compile을 다시 통과한다(호출부가 재검증 없이 저장할 수 있다는 계약).
    compile_strategy(parse_strategy_spec(draft))


# ---------------------------------------------------------------------------
# validator 재시도 — 컴파일 실패(지표 화이트리스트)
# ---------------------------------------------------------------------------


def test_draft_strategy_retries_after_unsupported_indicator_then_succeeds() -> None:
    bad_payload = {
        "strategy_name": "Bad",
        "version": "1.0",
        "universe": {"type": "single_asset", "tickers": [STOCK_CODE]},
        "entry": {"all": [{"left": "not_a_real_indicator", "operator": ">", "right": 0.2}]},
        "exit": {"any": [{"type": "max_holding_days", "value": 60}]},
        "execution": {"signal_time": "close", "trade_time": "next_open"},
    }
    client = FakeLlmClient([json.dumps(bad_payload, ensure_ascii=False), _valid_draft_json()])
    hypothesis = _approved_hypothesis()

    draft, metadata = draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=3,
    )

    assert metadata.num_attempts == 2
    assert len(client.calls) == 2
    retry_prompt = client.calls[1][1]
    assert "이전 응답의 문제" in retry_prompt
    assert "not_a_real_indicator" in retry_prompt
    assert draft["strategy_name"] == "TestDraft"


# ---------------------------------------------------------------------------
# validator 재시도 — tickers 강제
# ---------------------------------------------------------------------------


def test_draft_strategy_enforces_tickers_match_via_retry() -> None:
    wrong_ticker_json = _valid_draft_json(stock_code="999999")
    client = FakeLlmClient([wrong_ticker_json, _valid_draft_json()])
    hypothesis = _approved_hypothesis()

    draft, metadata = draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=3,
    )

    assert metadata.num_attempts == 2
    retry_prompt = client.calls[1][1]
    assert "이전 응답의 문제" in retry_prompt
    assert STOCK_CODE in retry_prompt
    assert draft["universe"] == {"type": "single_asset", "tickers": [STOCK_CODE]}


def test_draft_strategy_wrong_tickers_never_returned() -> None:
    """tickers 불일치로 소진되면 잘못된 초안이 반환되지 않고 예외로 끝난다."""
    wrong_ticker_json = _valid_draft_json(stock_code="999999")
    client = FakeLlmClient([wrong_ticker_json, wrong_ticker_json])
    hypothesis = _approved_hypothesis()

    with pytest.raises(DataValidationError):
        draft_strategy(
            hypothesis,
            stock_code=STOCK_CODE,
            client=client,
            prompts_dir=DEFAULT_PROMPTS_DIR,
            max_attempts=2,
        )


# ---------------------------------------------------------------------------
# validator 재시도 — strategy_name 공백 금지
# ---------------------------------------------------------------------------


def test_draft_strategy_enforces_nonblank_strategy_name_via_retry() -> None:
    blank_name_json = _valid_draft_json(strategy_name="")
    client = FakeLlmClient([blank_name_json, _valid_draft_json()])
    hypothesis = _approved_hypothesis()

    draft, metadata = draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=3,
    )

    assert metadata.num_attempts == 2
    assert "strategy_name" in client.calls[1][1]
    assert draft["strategy_name"] == "TestDraft"


# ---------------------------------------------------------------------------
# 재시도 소진
# ---------------------------------------------------------------------------


def test_draft_strategy_exhausts_attempts_and_raises() -> None:
    bad_payload = {
        "strategy_name": "Bad",
        "version": "1.0",
        "universe": {"type": "single_asset", "tickers": [STOCK_CODE]},
        "entry": {"all": [{"left": "not_a_real_indicator", "operator": ">", "right": 0.2}]},
        "exit": {"any": [{"type": "max_holding_days", "value": 60}]},
        "execution": {"signal_time": "close", "trade_time": "next_open"},
    }
    bad_json = json.dumps(bad_payload, ensure_ascii=False)
    client = FakeLlmClient([bad_json, bad_json, bad_json])
    hypothesis = _approved_hypothesis()

    with pytest.raises(DataValidationError, match="3회 시도"):
        draft_strategy(
            hypothesis,
            stock_code=STOCK_CODE,
            client=client,
            prompts_dir=DEFAULT_PROMPTS_DIR,
            max_attempts=3,
        )
    assert len(client.calls) == 3


def test_draft_strategy_rejects_non_dict_payload() -> None:
    client = FakeLlmClient(["[1, 2, 3]"])
    hypothesis = _approved_hypothesis()

    with pytest.raises(DataValidationError):
        draft_strategy(
            hypothesis,
            stock_code=STOCK_CODE,
            client=client,
            prompts_dir=DEFAULT_PROMPTS_DIR,
            max_attempts=1,
        )


# ---------------------------------------------------------------------------
# 프롬프트 렌더링 내용
# ---------------------------------------------------------------------------


def test_draft_strategy_prompt_includes_hypothesis_and_registry() -> None:
    hypothesis = _approved_hypothesis(thesis="유니크한논지문자열QWERTY")
    client = FakeLlmClient([_valid_draft_json()])

    draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=1,
    )

    assert len(client.calls) == 1
    _system_prompt, user_prompt = client.calls[0]
    assert "유니크한논지문자열QWERTY" in user_prompt
    assert STOCK_CODE in user_prompt
    assert "operating_income_yoy" in user_prompt  # 재무지표 화이트리스트
    assert "sma_20" in user_prompt  # 가격지표 화이트리스트
    assert "foreign_net_buy_20d" in user_prompt  # 수급지표 화이트리스트
    assert '"signal_time": "close"' in user_prompt  # 예시 JSON 인라인
    # 렌더링되지 않은 플레이스홀더가 남아있지 않아야 한다(kwargs 누락 조기 발견).
    assert "{stock_code}" not in user_prompt
    assert "{thesis}" not in user_prompt


# ---------------------------------------------------------------------------
# 예시 JSON 상수 — README §23.4 fixture와의 드리프트 방지
# ---------------------------------------------------------------------------


def test_example_strategy_spec_matches_fixture_and_compiles(
    earnings_flow_breakout_raw: dict[str, Any],
) -> None:
    assert earnings_flow_breakout_raw == _EXAMPLE_STRATEGY_SPEC
    compile_strategy(parse_strategy_spec(_EXAMPLE_STRATEGY_SPEC))
