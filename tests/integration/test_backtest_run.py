"""백테스트 실데이터 end-to-end integration (명세 A6 §6·§7).

DATA_DIR 환경변수의 실데이터(A3 normalized market + A4 financial_metrics)로 §23
기본 전략을 실행한다. 데이터가 없으면 skip한다. **거래 수·성과는 assert하지
않고 관찰만 한다**(전략 성과는 검증 대상이 아니라 관찰 대상, 명세 A6 §6).

- 2016-01-01~2025-12-31 전체: 예외 없이 완료 + 산출물 3종 + assert_no_lookahead
  통과 + B&H·KOSPI 비교 산출. 재무 metrics가 2021~ 이므로 2016~2020은 무포지션이
  정상이다(명세 A6 §6).
- 2021~2025 부분 구간도 함께 실행한다.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from research_backtest.core.hitl.models import StrategyReview
from research_backtest.quant.backtest.costs import load_backtest_config
from research_backtest.quant.backtest.data import assert_no_lookahead, build_backtest_frame
from research_backtest.quant.backtest.runner import (
    BACKTEST_RESULT_FILENAME,
    DAILY_PORTFOLIO_FILENAME,
    TRADE_LOG_FILENAME,
    _load_daily,
    _load_metrics,
    execute_approved_strategy,
)

pytestmark = pytest.mark.integration

STOCK = "000660"
CORP = "00164779"
FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "strategy" / "earnings_flow_breakout.json"
)
REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "backtest.yaml"


@pytest.fixture
def data_dir() -> Path:
    """DATA_DIR 환경변수의 실데이터 디렉토리 — 없거나 파일 미비 시 skip."""
    raw = os.environ.get("DATA_DIR")
    if not raw:
        pytest.skip("DATA_DIR 미설정 — 실데이터 백테스트 integration 생략 (명세 A6 §7)")
    root = Path(raw)
    daily = root / "normalized" / "market" / STOCK / "daily.parquet"
    metrics = root / "normalized" / "financials" / CORP / "financial_metrics.parquet"
    if not (daily.exists() and metrics.exists()):
        pytest.skip(f"실데이터 parquet 없음({daily}·{metrics}) — integration 생략")
    return root


@pytest.fixture
def approved_review() -> StrategyReview:
    strategy = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return StrategyReview(
        review_id="rv-int",
        hypothesis_id="hyp-int",
        llm_draft_strategy=strategy,
        final_strategy=strategy,
        modifications=[],
        approval_reason="integration",
        approved_by="integration-tester",
        approved_at="2026-07-14T09:00:00+09:00",
    )


def _run(data_dir: Path, review: StrategyReview, out: Path, start: date, end: date):  # type: ignore[no-untyped-def]
    return execute_approved_strategy(
        review,
        data_dir=data_dir,
        stock_code=STOCK,
        corp_code=CORP,
        start_date=start,
        end_date=end,
        out_dir=out,
        backtest_config=load_backtest_config(REPO_CONFIG),
    )


def test_full_period_2016_2025(
    data_dir: Path, approved_review: StrategyReview, tmp_path: Path
) -> None:
    """2016~2025 전체 — 예외 없이 완료, 산출물 3종, B&H·KOSPI 비교 산출."""
    out = tmp_path / "full"
    result = _run(data_dir, approved_review, out, date(2016, 1, 1), date(2025, 12, 31))

    assert (out / BACKTEST_RESULT_FILENAME).exists()
    assert (out / TRADE_LOG_FILENAME).exists()
    assert (out / DAILY_PORTFOLIO_FILENAME).exists()
    assert result.trading_days > 0
    # B&H·KOSPI 비교가 산출된다(값 자체는 관찰 대상)
    assert result.buy_hold.cumulative_return is not None
    assert result.benchmark.cumulative_return is not None
    assert result.benchmark.name == "KOSPI"


def test_partial_period_2021_2025(
    data_dir: Path, approved_review: StrategyReview, tmp_path: Path
) -> None:
    """2021~2025 부분 구간 — 재무 신호가 유효한 구간."""
    out = tmp_path / "partial"
    result = _run(data_dir, approved_review, out, date(2021, 1, 1), date(2025, 12, 31))
    assert result.trading_days > 0
    assert result.buy_hold.cumulative_return is not None


def test_assert_no_lookahead_on_real_join(data_dir: Path) -> None:
    """실데이터 as-of join이 룩어헤드 방어 검증을 통과한다(명세 A6 §7)."""
    daily = _load_daily(data_dir, STOCK)
    metrics = _load_metrics(data_dir, CORP)
    joined = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=date(2016, 1, 1), end_date=date(2025, 12, 31)
    )
    assert_no_lookahead(joined)  # 위반 시 LookaheadError


def test_pre_financial_period_has_no_positions(
    data_dir: Path, approved_review: StrategyReview, tmp_path: Path
) -> None:
    """재무 metrics 부재 구간(2016~2020)은 무포지션이 정상이다(명세 A6 §6)."""
    out = tmp_path / "prefinancial"
    result = _run(data_dir, approved_review, out, date(2016, 1, 1), date(2020, 12, 31))
    # operating_income_yoy NaN → entry False → 무거래
    assert result.num_trades == 0
    assert result.has_trades is False
