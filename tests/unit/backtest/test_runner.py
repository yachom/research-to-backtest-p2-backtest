"""runner.py — 게이트 강제·산출물 3종 저장 테스트 (명세 A6 §5·§6, 1804 §8·§13).

미승인 review는 ApprovalGateError로 막고 산출물을 만들지 않는다. 승인 review는
backtest_result.json·trade_log.csv·daily_portfolio.csv를 생성한다. 오프라인 —
tmp_path에 소형 parquet 데이터셋을 만들어 파일 I/O 경로까지 통과시킨다.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from research_backtest.core.exceptions import ApprovalGateError
from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.runner import (
    BACKTEST_RESULT_FILENAME,
    DAILY_PORTFOLIO_FILENAME,
    TRADE_LOG_FILENAME,
    execute_approved_strategy,
)

STOCK = "000660"
CORP = "00164779"
INDEX = "1001"

SIMPLE_STRATEGY: dict[str, object] = {
    "strategy_name": "SimplePriceTest",
    "universe": {"type": "single_asset", "tickers": [STOCK]},
    "entry": {"all": [{"left": "close", "operator": ">", "right": "sma_5"}]},
    "exit": {"any": [{"type": "max_holding_days", "value": 3}]},
}

TEST_CONFIG = BacktestConfig(
    commission_rate=0.00015, sell_tax_rate=0.0018, slippage_rate=0.001, initial_cash=10_000_000.0
)


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _build_data_dir(tmp_path: Path) -> Path:
    """A3/A4 정규화 산출 스키마의 소형 parquet 데이터셋을 만든다."""
    data_dir = tmp_path / "data"
    dates = _weekdays(date(2024, 1, 1), 30)
    # 진입·청산이 생기도록 오르내리는 종가
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


# --- 미승인 게이트 (명세 A6 §5·§6, 1804 §13) ---------------------------------


def test_unapproved_review_raises_and_writes_nothing(tmp_path: Path) -> None:
    """미승인(review=None) → ApprovalGateError, 산출물 미생성."""
    data_dir = _build_data_dir(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(ApprovalGateError):
        execute_approved_strategy(
            None,  # type: ignore[arg-type]  # 승인 기록 없음 = 미승인
            data_dir=data_dir,
            stock_code=STOCK,
            corp_code=CORP,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 15),
            out_dir=out_dir,
            backtest_config=TEST_CONFIG,
        )
    assert not out_dir.exists()  # 산출물 디렉토리 자체가 생기지 않았다


def test_strategy_review_requires_nonblank_approver() -> None:
    """승인 주체(approved_by)가 공백인 review는 애초에 만들 수 없다 — 게이트의 전제."""
    from pydantic import ValidationError

    from research_backtest.core.hitl.models import StrategyReview

    with pytest.raises(ValidationError):
        StrategyReview(
            review_id="r",
            hypothesis_id="h",
            llm_draft_strategy={},
            final_strategy={},
            modifications=[],
            approval_reason="x",
            approved_by="  ",
            approved_at="t",
        )


# --- 승인 → 산출물 3종 (명세 A6 §5·§6) --------------------------------------


def test_approved_review_writes_three_artifacts(tmp_path: Path, approved_review) -> None:  # type: ignore[no-untyped-def]
    data_dir = _build_data_dir(tmp_path)
    out_dir = tmp_path / "out"
    review = approved_review(SIMPLE_STRATEGY)

    result = execute_approved_strategy(
        review,
        data_dir=data_dir,
        stock_code=STOCK,
        corp_code=CORP,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 15),
        out_dir=out_dir,
        backtest_config=TEST_CONFIG,
    )

    assert (out_dir / BACKTEST_RESULT_FILENAME).exists()
    assert (out_dir / TRADE_LOG_FILENAME).exists()
    assert (out_dir / DAILY_PORTFOLIO_FILENAME).exists()
    assert result.strategy_name == "SimplePriceTest"
    assert result.trading_days > 0
    # 산출물 JSON이 결과와 일치
    import json

    saved = json.loads((out_dir / BACKTEST_RESULT_FILENAME).read_text())
    assert saved["num_trades"] == result.num_trades
    assert saved["has_trades"] == result.has_trades


def test_approved_review_revalidates_strategy(tmp_path: Path, approved_review) -> None:  # type: ignore[no-untyped-def]
    """승인본이 DSL 규칙 위반이면 재검증에서 실패한다(명세 A6 §5)."""
    from research_backtest.core.exceptions import StrategyValidationError

    data_dir = _build_data_dir(tmp_path)
    bad = dict(SIMPLE_STRATEGY)
    bad["entry"] = {"all": [{"left": "not_a_real_indicator", "operator": ">", "right": 1.0}]}
    review = approved_review(bad)
    with pytest.raises(StrategyValidationError):
        execute_approved_strategy(
            review,
            data_dir=data_dir,
            stock_code=STOCK,
            corp_code=CORP,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 15),
            out_dir=tmp_path / "out",
            backtest_config=TEST_CONFIG,
        )
