"""``generate-strategy-draft`` CLI 단위 테스트 (명세 W3b-candidates-strategy.md §3.2, §4).

``research_backtest.app.cli``는 건드리지 않는다 — ``typer.Typer()`` 새
인스턴스에 ``hitl_flow.register(app)``로 구성해 테스트한다(기존
tests/unit/test_cli_hitl.py와 동일한 관례, 명세 §1 "두 트랙 각자 자기 구역
헬퍼로 — 공유 금지"에 따라 헬퍼는 이 파일에 독립적으로 둔다).

LLM 주입은 ``hitl_flow.create_llm_client``\\ (모듈 수준에 임포트된 팩토리
참조)를 :class:`FakeLlmClient`\\ 를 반환하는 함수로 monkeypatch해서
수행한다 — ``generate_strategy_draft``가 함수 내부에서 클라이언트를
생성하므로 이 방법이 유일한 주입 지점이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from research_backtest.app.commands import hitl_flow
from research_backtest.core.config import Settings
from research_backtest.core.hitl.models import HumanInvestmentHypothesis, RunManifest
from research_backtest.core.hitl.states import (
    FORWARD_ORDER,
    PipelineState,
    advance,
    create_run_state,
)
from research_backtest.core.hitl.store import RunStore
from research_backtest.core.llm.testing import FakeLlmClient

runner = CliRunner()

STOCK_CODE = "000660"


def _build_app() -> typer.Typer:
    application = typer.Typer()
    hitl_flow.register(application)
    return application


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """rich 표가 비-TTY 테스트 환경(기본 width=80)에서 잘리지 않도록 폭을 넓힌다."""
    monkeypatch.setattr(hitl_flow, "console", Console(width=200))


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        dart_api_key="unit-test-key",
        claude_code_oauth_token="",
        anthropic_api_key="",
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
    )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    active = _make_settings(tmp_path)
    monkeypatch.setattr(hitl_flow, "get_settings", lambda: active)
    return active


def _make_run(
    outputs_dir: Path,
    run_id: str,
    *,
    state: PipelineState,
    stock_code: str = STOCK_CODE,
    company: str = "SK하이닉스",
    as_of_date: str = "2025-12-31",
) -> RunStore:
    """지정한 ``state``까지 정방향으로 전진시킨 run_state.json + run_manifest.json 픽스처.

    ``generate_strategy_draft``\\ 는 ``run_manifest.json``\\ 에서 ``stock_code``를
    읽으므로(§3.2), tests/unit/test_cli_hitl.py의 ``_make_run``과 달리 manifest도
    함께 저장한다.
    """
    store = RunStore(outputs_dir, run_id)
    run_state = create_run_state(run_id, company, as_of_date, actor="test-fixture")
    target_idx = FORWARD_ORDER.index(state)
    for target in FORWARD_ORDER[1 : target_idx + 1]:
        run_state = advance(run_state, target, actor="test-fixture")
    store.save_run_state(run_state)
    store.save_run_manifest(
        RunManifest(
            run_id=run_id,
            company_query=company,
            corp_code="00164779",
            corp_name=company,
            corp_eng_name="SK hynix Inc.",
            stock_code=stock_code,
            as_of_date=as_of_date,
            created_at="2026-07-14T09:00:00+09:00",
        )
    )
    return store


def _approved_hypothesis_payload(**overrides: Any) -> dict[str, Any]:
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
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-07-14T12:00:00+09:00",
        "created_at": "2026-07-14T11:00:00+09:00",
        "updated_at": "2026-07-14T11:00:00+09:00",
    }
    payload.update(overrides)
    return payload


def _valid_draft_response(*, stock_code: str = STOCK_CODE, strategy_name: str = "TestDraft") -> str:
    payload = {
        "strategy_name": strategy_name,
        "version": "1.0",
        "universe": {"type": "single_asset", "tickers": [stock_code]},
        "entry": {"all": [{"left": "operating_income_yoy", "operator": ">", "right": 0.2}]},
        "exit": {"any": [{"type": "max_holding_days", "value": 60}]},
        "execution": {"signal_time": "close", "trade_time": "next_open"},
    }
    return json.dumps(payload, ensure_ascii=False)


def _patch_llm_client(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> FakeLlmClient:
    client = FakeLlmClient(responses)
    monkeypatch.setattr(hitl_flow, "create_llm_client", lambda config, settings: client)
    return client


def _save_approved_hypothesis(store: RunStore, **overrides: Any) -> None:
    store.save_human_hypothesis(
        HumanInvestmentHypothesis.model_validate(_approved_hypothesis_payload(**overrides))
    )


# ---------------------------------------------------------------------------
# 상태 정책 분기 1 — 이전 단계 미도달(기존 게이트, exit 4)
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_hypothesis_not_reached_exits_4(settings: Settings) -> None:
    _make_run(settings.outputs_dir, "RUN-GSD1", state=PipelineState.HYPOTHESIS_DRAFT)
    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD1"])
    assert result.exit_code == 4


# ---------------------------------------------------------------------------
# 상태 정책 분기 2 — 정상 경로(전진 2회) + AIUsageRecord
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_success_advances_twice_and_records_usage(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD2", state=PipelineState.HYPOTHESIS_APPROVED)
    _save_approved_hypothesis(store)
    client = _patch_llm_client(monkeypatch, [_valid_draft_response()])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD2"])
    assert result.exit_code == 0, result.output
    assert "strategy_draft 저장 완료" in result.output
    assert "approve-strategy --review" in result.output
    assert "파이프라인 상태: AWAITING_STRATEGY_REVIEW" in result.output

    run_state = store.load_run_state()
    assert run_state.current_state == PipelineState.AWAITING_STRATEGY_REVIEW
    assert run_state.transitions[-2].to_state == PipelineState.STRATEGY_DRAFT_READY
    assert run_state.transitions[-2].actor == "system"
    assert run_state.transitions[-1].to_state == PipelineState.AWAITING_STRATEGY_REVIEW
    assert run_state.transitions[-1].actor == "system"

    draft = store.load_strategy_draft()
    assert draft["universe"] == {"type": "single_asset", "tickers": [STOCK_CODE]}

    usage = store.load_ai_usage_log()
    assert len(usage) == 1
    record = usage[0]
    assert record.stage == "strategy_translation"
    assert record.model == "fake"
    assert record.prompt_name == "strategy_translation"
    assert record.prompt_version == "v1"
    assert record.input_artifact_ids == ["human_investment_hypothesis.json"]
    assert record.output_artifact_ids == ["strategy_draft.json"]
    assert record.ai_role == "전략 초안 변환"
    assert record.human_review_required is True
    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# 상태 정책 분기 3 — 재생성(전이 없음)
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_regeneration_at_strategy_draft_ready_no_transition(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD3", state=PipelineState.STRATEGY_DRAFT_READY)
    _save_approved_hypothesis(store)
    store.save_strategy_draft(json.loads(_valid_draft_response(strategy_name="Old")))
    before = store.load_run_state()
    _patch_llm_client(monkeypatch, [_valid_draft_response(strategy_name="New")])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD3"])
    assert result.exit_code == 0, result.output
    assert "재생성 — 상태 전이 없음" in result.output

    after = store.load_run_state()
    assert after.current_state == PipelineState.STRATEGY_DRAFT_READY
    assert len(after.transitions) == len(before.transitions)
    assert store.load_strategy_draft()["strategy_name"] == "New"


def test_generate_strategy_draft_regeneration_at_awaiting_review_no_transition(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(
        settings.outputs_dir, "RUN-GSD4", state=PipelineState.AWAITING_STRATEGY_REVIEW
    )
    _save_approved_hypothesis(store)
    store.save_strategy_draft(json.loads(_valid_draft_response(strategy_name="Old")))
    before = store.load_run_state()
    _patch_llm_client(monkeypatch, [_valid_draft_response(strategy_name="New")])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD4"])
    assert result.exit_code == 0, result.output
    assert "재생성 — 상태 전이 없음" in result.output

    after = store.load_run_state()
    assert after.current_state == PipelineState.AWAITING_STRATEGY_REVIEW
    assert len(after.transitions) == len(before.transitions)
    assert store.load_strategy_draft()["strategy_name"] == "New"


# ---------------------------------------------------------------------------
# 상태 정책 분기 4 — 승인본 무효화 방지(exit 4, LLM 호출 이전에 거부)
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_strategy_approved_exits_4(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD5", state=PipelineState.STRATEGY_APPROVED)
    _save_approved_hypothesis(store)
    client = _patch_llm_client(monkeypatch, [_valid_draft_response()])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD5"])
    assert result.exit_code == 4
    assert client.calls == []  # 거부는 LLM 호출 전에 일어나야 한다(불필요한 과금 방지)


def test_generate_strategy_draft_complete_state_exits_4(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD5B", state=PipelineState.COMPLETE)
    _save_approved_hypothesis(store)
    client = _patch_llm_client(monkeypatch, [_valid_draft_response()])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD5B"])
    assert result.exit_code == 4
    assert client.calls == []


# ---------------------------------------------------------------------------
# LLM 인증 미설정 — exit 3 (ConfigError, 실제 create_llm_client 경로)
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_llm_auth_missing_exits_3(settings: Settings) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD6", state=PipelineState.HYPOTHESIS_APPROVED)
    _save_approved_hypothesis(store)

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD6"])
    assert result.exit_code == 3
    assert "설정 오류" in result.output


# ---------------------------------------------------------------------------
# 재시도 경로가 CLI 전체 배선을 통해서도 동작하는지
# ---------------------------------------------------------------------------


def test_generate_strategy_draft_retries_end_to_end_through_cli(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_run(settings.outputs_dir, "RUN-GSD7", state=PipelineState.HYPOTHESIS_APPROVED)
    _save_approved_hypothesis(store)
    wrong_ticker = _valid_draft_response(stock_code="999999")
    client = _patch_llm_client(monkeypatch, [wrong_ticker, _valid_draft_response()])

    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD7"])
    assert result.exit_code == 0, result.output
    assert len(client.calls) == 2
    assert store.load_strategy_draft()["universe"] == {
        "type": "single_asset",
        "tickers": [STOCK_CODE],
    }


def test_generate_strategy_draft_missing_hypothesis_file_exits_1(settings: Settings) -> None:
    # 상태는 HYPOTHESIS_APPROVED(게이트 통과)이지만 human_investment_hypothesis.json이
    # 없는 경우 — DataValidationError → exit 1(§3 오류 매핑).
    _make_run(settings.outputs_dir, "RUN-GSD8", state=PipelineState.HYPOTHESIS_APPROVED)
    result = runner.invoke(_build_app(), ["generate-strategy-draft", "--run-id", "RUN-GSD8"])
    assert result.exit_code == 1
