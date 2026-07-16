"""전략 초안 생성기 실호출 integration 테스트 (명세 docs/specs/W3b-candidates-strategy.md §3.3) —
인증 없으면 skip.

실행: 레포 루트에서 메인 레포 .env를 주입하고 ``pytest -m integration``
(또는 이 파일만 ``pytest tests/integration/test_strategy_draft_live.py``).
구독 계정의 rate limit을 고려해 실 호출은 1회로 최소화한다
(tests/integration/test_llm_live.py와 동일한 관례).
"""

from __future__ import annotations

import pytest

from research_backtest.core.config import get_settings
from research_backtest.core.hitl.models import HumanInvestmentHypothesis
from research_backtest.core.llm.client import LlmTextClient, create_llm_client
from research_backtest.core.llm.config import LlmConfig, load_llm_config
from research_backtest.quant.strategy.compiler import compile_strategy
from research_backtest.quant.strategy.draft import DEFAULT_PROMPTS_DIR, draft_strategy
from research_backtest.quant.strategy.schema import parse_strategy_spec

pytestmark = pytest.mark.integration

STOCK_CODE = "000660"


def _approved_hypothesis() -> HumanInvestmentHypothesis:
    """README §23.4 변수(operating_income_yoy 등)를 사용하는 승인 가설 픽스처."""
    return HumanInvestmentHypothesis.model_validate(
        {
            "hypothesis_id": "HYP-LIVE-1",
            "view_id": "VIEW-LIVE-1",
            "author": "테스트",
            "thesis": "영업이익 개선과 외국인 수급 유입이 겹치는 구간에서 가격 돌파가 이어진다.",
            "economic_rationale": "실적 서프라이즈가 수급 개선으로 이어져 추세를 형성한다.",
            "expected_mechanism": "영업이익 YoY 개선 → 외국인 순매수 유입 → 가격 돌파",
            "selected_variables": [
                "operating_income_yoy",
                "foreign_net_buy_20d",
                "rolling_high_60",
            ],
            "expected_direction": "up",
            "investment_horizon_days": 60,
            "evidence_ids": ["EVID-001"],
            "falsification_conditions": ["2개 분기 연속 영업이익 컨센서스 하회 시 기각"],
            "limitations": [],
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-07-15T10:00:00+09:00",
            "created_at": "2026-07-15T09:00:00+09:00",
            "updated_at": "2026-07-15T09:00:00+09:00",
        }
    )


@pytest.fixture(scope="module")
def llm_config() -> LlmConfig:
    return load_llm_config()


@pytest.fixture(scope="module")
def llm_client(llm_config: LlmConfig) -> LlmTextClient:
    settings = get_settings()
    if not settings.anthropic_api_key and not settings.claude_code_oauth_token:
        pytest.skip(
            "LLM 인증(CLAUDE_CODE_OAUTH_TOKEN 또는 ANTHROPIC_API_KEY) 미설정 — "
            "integration 테스트 생략(명세 W3a-llm-evidence.md §2.6)"
        )
    return create_llm_client(llm_config, settings)


def test_draft_strategy_live_call_parses_compiles_and_enforces_tickers(
    llm_client: LlmTextClient, llm_config: LlmConfig
) -> None:
    """실호출 1회(재시도 포함 가능) — 승인 가설 → 전략 초안이 parse+compile 통과,
    tickers·execution이 강제된 값과 일치하는지 확인한다(명세 §3.3)."""
    hypothesis = _approved_hypothesis()

    draft, metadata = draft_strategy(
        hypothesis,
        stock_code=STOCK_CODE,
        client=llm_client,
        prompts_dir=DEFAULT_PROMPTS_DIR,
        max_attempts=llm_config.max_attempts,
    )

    spec = parse_strategy_spec(draft)
    compile_strategy(spec)  # 지표 화이트리스트까지 재확인 — 성공만으로 충분히 검증된다

    assert spec.universe.tickers == [STOCK_CODE]
    assert spec.execution.signal_time == "close"
    assert spec.execution.trade_time == "next_open"
    assert spec.strategy_name.strip() != ""
    assert metadata.num_attempts >= 1
    assert metadata.model.startswith(llm_config.model)
