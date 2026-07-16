"""backtest CLI 단위 테스트 — 게이트·전이·재실행·산출물 (명세 §4.4·§7).

승인 게이트가 실제로 강제되는지(미승인→exit 4), 상태 전이 2회(STRATEGY_APPROVED→
BACKTEST_COMPLETE→AWAITING_INTERPRETATION), 재실행 시 전이 없음, 산출물 3종 존재를
검증한다. 합성 daily/metrics/index parquet은 tests/unit/backtest/test_runner.py의
픽스처 패턴을 재사용한다(수정 없이 동일 구성 복제). 수치 검증은 A6 소관이므로 반복하지 않는다.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from research_backtest.app import cli
from research_backtest.app.commands import backtest_cmd
from research_backtest.core.config import Settings
from research_backtest.core.hitl.models import (
    HumanInvestmentHypothesis,
    HypothesisStatus,
    StrategyReview,
)
from research_backtest.core.hitl.states import PipelineState, RunState, StateTransition
from research_backtest.core.hitl.store import RunStore
from research_backtest.quant.backtest.runner import (
    BACKTEST_RESULT_FILENAME,
    DAILY_PORTFOLIO_FILENAME,
    TRADE_LOG_FILENAME,
)

runner = CliRunner()

RUN_ID = "20260715_090000_SK_HYNIX"
STOCK = "000660"
CORP = "00164779"
INDEX = "1001"
AS_OF = "2024-02-15"

SIMPLE_STRATEGY: dict[str, object] = {
    "strategy_name": "SimplePriceTest",
    "universe": {"type": "single_asset", "tickers": [STOCK]},
    "entry": {"all": [{"left": "close", "operator": ">", "right": "sma_5"}]},
    "exit": {"any": [{"type": "max_holding_days", "value": 3}]},
}


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _build_data_dir(root: Path) -> Path:
    """A3/A4 정규화 산출 스키마의 소형 parquet 데이터셋 (test_runner.py 패턴 복제)."""
    data_dir = root / "data"
    dates = _weekdays(date(2024, 1, 1), 30)
    closes = [100 + (10 if i % 6 < 3 else -5) + i for i in range(30)]
    opens = [c - 1 for c in closes]

    stock_dir = data_dir / "normalized" / "market" / STOCK
    stock_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [c + 2 for c in closes],
            "low": [o - 2 for o in opens],
            "close": closes,
            "volume": [1000] * 30,
            "foreign_net_buy_value": [0] * 30,
            "institution_net_buy_value": [0] * 30,
        }
    ).to_parquet(stock_dir / "daily.parquet", engine="pyarrow", index=False)

    index_dir = data_dir / "normalized" / "market" / f"index_{INDEX}"
    index_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": [2000.0] * 30,
            "high": [2010.0] * 30,
            "low": [1990.0] * 30,
            "close": [2000.0 + i for i in range(30)],
            "volume": [1] * 30,
            "trading_value": [1] * 30,
        }
    ).to_parquet(index_dir / "daily.parquet", engine="pyarrow", index=False)

    fin_dir = data_dir / "normalized" / "financials" / CORP
    fin_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "metric_id": ["operating_income_yoy"],
            "fs_scope": ["CFS"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "period_end": [date(2024, 3, 31)],
            "value": [0.5],
            "rcept_no": ["r1"],
            "rcept_dt": [date(2024, 1, 10)],
            "available_from": [date(2024, 1, 11)],
            "inputs_derived": [False],
        }
    ).to_parquet(fin_dir / "financial_metrics.parquet", engine="pyarrow", index=False)
    return data_dir


def _run_state(state: PipelineState) -> RunState:
    """지정 상태의 RunState — 전이 이력은 최초 진입 1건만(command는 current_state만 읽는다)."""
    return RunState(
        run_id=RUN_ID,
        company="SK하이닉스",
        as_of_date=AS_OF,
        current_state=state,
        transitions=[
            StateTransition(
                from_state=None,
                to_state=PipelineState.DATA_READY,
                actor="test-fixture",
                at="2026-07-15T09:00:00+09:00",
                auto_approved=False,
                note=None,
            )
        ],
    )


def _hypothesis(hypothesis_id: str = "hyp-1") -> HumanInvestmentHypothesis:
    return HumanInvestmentHypothesis(
        hypothesis_id=hypothesis_id,
        view_id="view-1",
        author="tester",
        thesis="영업이익 개선이 주가를 견인한다.",
        economic_rationale="수요 회복.",
        expected_mechanism="이익 개선 → 재평가.",
        selected_variables=["operating_income_yoy"],
        expected_direction="POSITIVE",
        investment_horizon_days=180,
        evidence_ids=["ev-1"],
        falsification_conditions=["영업이익 개선에도 주가 하락"],
        limitations=[],
        status=HypothesisStatus.APPROVED,
        created_at="2026-07-15T08:00:00+09:00",
        updated_at="2026-07-15T08:30:00+09:00",
        approved_by="tester",
        approved_at="2026-07-15T08:30:00+09:00",
    )


def _review(hypothesis_id: str = "hyp-1") -> StrategyReview:
    return StrategyReview(
        review_id="rv-1",
        hypothesis_id=hypothesis_id,
        llm_draft_strategy=SIMPLE_STRATEGY,
        final_strategy=SIMPLE_STRATEGY,
        modifications=[],
        approval_reason="테스트 승인",
        approved_by="tester",
        approved_at="2026-07-15T09:00:00+09:00",
    )


def _write_manifest(store: RunStore, *, include: bool = True) -> None:
    if not include:
        return
    payload = {
        "run_id": RUN_ID,
        "company_query": "SK하이닉스",
        "corp_code": CORP,
        "corp_name": "SK하이닉스",
        "corp_eng_name": "SK hynix Inc.",
        "stock_code": STOCK,
        "as_of_date": AS_OF,
        "created_at": "2026-07-15T09:00:00+09:00",
        "code_version": None,
    }
    store.run_dir.mkdir(parents=True, exist_ok=True)
    (store.run_dir / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: PipelineState = PipelineState.STRATEGY_APPROVED,
    manifest: bool = True,
    run_state: bool = True,
    hypothesis: bool = True,
    review: bool = True,
    review_hyp_id: str = "hyp-1",
) -> tuple[Settings, RunStore]:
    """settings 주입 + run 산출물 픽스처 구성."""
    data_dir = _build_data_dir(tmp_path)
    settings = Settings(
        _env_file=None,
        dart_api_key="unit-test-key",
        data_dir=data_dir,
        outputs_dir=tmp_path / "outputs",
    )
    monkeypatch.setattr(backtest_cmd, "get_settings", lambda: settings)
    store = RunStore(settings.outputs_dir, RUN_ID)
    if run_state:
        store.save_run_state(_run_state(state))
    _write_manifest(store, include=manifest)
    if hypothesis:
        store.save_human_hypothesis(_hypothesis())
    if review:
        store.save_strategy_review(_review(review_hyp_id))
    return settings, store


# --- 정상 경로 + 전이 (명세 §4.4 절차 6) -------------------------------------


def test_backtest_happy_path_transitions_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STRATEGY_APPROVED → 실행 → BACKTEST_COMPLETE → AWAITING_INTERPRETATION (전이 2회)."""
    _settings, store = _setup(tmp_path, monkeypatch)
    result = runner.invoke(
        cli.app,
        ["backtest", "--run-id", RUN_ID, "--start-date", "2024-01-01", "--end-date", "2024-02-15"],
    )
    assert result.exit_code == 0, result.output
    # 산출물 3종
    assert (store.run_dir / BACKTEST_RESULT_FILENAME).exists()
    assert (store.run_dir / TRADE_LOG_FILENAME).exists()
    assert (store.run_dir / DAILY_PORTFOLIO_FILENAME).exists()
    # 상태 전이 확인 (run_state.json 재로드)
    reloaded = store.load_run_state()
    assert reloaded.current_state == PipelineState.AWAITING_INTERPRETATION
    assert reloaded.transitions[-1].actor == "system"
    assert reloaded.transitions[-1].auto_approved is False
    assert reloaded.transitions[-2].to_state == PipelineState.BACKTEST_COMPLETE
    # 상태 표시 2줄 (명세 §3)
    assert "파이프라인 상태: AWAITING_INTERPRETATION" in result.output
    assert "다음 단계: submit-interpretation" in result.output


def test_backtest_uses_manifest_as_of_for_end_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--end-date 미지정 시 run_manifest.as_of_date를 종료일로 쓴다 (명세 §4.4)."""
    _settings, store = _setup(tmp_path, monkeypatch)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID, "--start-date", "2024-01-01"])
    assert result.exit_code == 0, result.output
    saved = json.loads((store.run_dir / BACKTEST_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert saved["end_date"] == AS_OF


# --- 게이트 (명세 §4.4 절차 3·4, CLAUDE.md §3) -------------------------------


def test_backtest_below_strategy_approved_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """STRATEGY_APPROVED 미달 상태는 게이트 차단(exit 4)."""
    _settings, store = _setup(tmp_path, monkeypatch, state=PipelineState.DATA_READY)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 4, result.output
    # 산출물 미생성
    assert not (store.run_dir / BACKTEST_RESULT_FILENAME).exists()


def test_backtest_complete_state_exits_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """해석까지 완료(COMPLETE)된 실행은 재백테스트 거부(exit 4)."""
    _setup(tmp_path, monkeypatch, state=PipelineState.COMPLETE)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 4, result.output
    assert "새 run" in result.output


def test_backtest_unapproved_hypothesis_exits_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """가설이 미승인(DRAFT)이면 가설 게이트 차단(exit 4)."""
    _settings, store = _setup(tmp_path, monkeypatch, hypothesis=False)
    draft = _hypothesis().model_copy(
        update={"status": HypothesisStatus.DRAFT, "approved_by": None, "approved_at": None}
    )
    store.save_human_hypothesis(draft)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 4, result.output
    assert not (store.run_dir / BACKTEST_RESULT_FILENAME).exists()


def test_backtest_hypothesis_id_mismatch_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """전략 리뷰의 hypothesis_id가 승인 가설과 다르면 exit 1."""
    _setup(tmp_path, monkeypatch, review_hyp_id="hyp-OTHER")
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 1, result.output
    assert "hypothesis_id" in result.output


# --- 부재 경로 (명세 §4.4 절차 1·2) ------------------------------------------


def test_backtest_missing_run_state_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_state.json 부재는 exit 1 + create-run 안내."""
    _setup(tmp_path, monkeypatch, run_state=False, manifest=False, hypothesis=False, review=False)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 1, result.output
    assert "create-run" in result.output


def test_backtest_missing_manifest_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_manifest.json 부재는 exit 1 + create-run 안내."""
    _setup(tmp_path, monkeypatch, manifest=False)
    result = runner.invoke(cli.app, ["backtest", "--run-id", RUN_ID])
    assert result.exit_code == 1, result.output
    assert "create-run" in result.output


# --- 재실행 (명세 §4.4 절차 6) -----------------------------------------------


def test_backtest_rerun_no_transition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AWAITING_INTERPRETATION에서의 재실행은 산출물만 갱신, 상태 전이 없음."""
    _settings, store = _setup(tmp_path, monkeypatch, state=PipelineState.AWAITING_INTERPRETATION)
    result = runner.invoke(
        cli.app,
        ["backtest", "--run-id", RUN_ID, "--start-date", "2024-01-01", "--end-date", "2024-02-15"],
    )
    assert result.exit_code == 0, result.output
    assert "재실행 — 상태 전이 없음" in result.output
    reloaded = store.load_run_state()
    assert reloaded.current_state == PipelineState.AWAITING_INTERPRETATION
    # 전이가 추가되지 않았다 (초기 1건 그대로)
    assert len(reloaded.transitions) == 1
    assert (store.run_dir / BACKTEST_RESULT_FILENAME).exists()


def test_backtest_benchmark_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--benchmark 지정 시 config를 model_copy로 갱신해 결과에 반영한다."""
    _settings, store = _setup(tmp_path, monkeypatch)
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--run-id",
            RUN_ID,
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-02-15",
            "--benchmark",
            "KOSDAQ",
        ],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads((store.run_dir / BACKTEST_RESULT_FILENAME).read_text(encoding="utf-8"))
    assert saved["benchmark"]["name"] == "KOSDAQ"
