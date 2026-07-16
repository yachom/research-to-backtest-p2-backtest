"""게이트 강제 진입점 + 산출물 저장 (명세 A6 §5, 1804 §8·§13, AI_ROLE_BOUNDARY §3).

:func:`execute_approved_strategy`는 **첫 줄에서** 승인 게이트를 검사한다 —
승인 기록(:class:`StrategyReview`) 없이 백테스트를 실행할 수 없는 유일한 공식
진입점이다(1804 §13 "승인되지 않은 단계를 자동으로 건너뛰어 백테스트하지
않도록 한다"). :func:`engine.run_backtest`를 직접 부르는 것은 테스트·연구용이며
게이트를 우회한다.

파이프라인(명세 A6 §2 순서): financial as-of join → ``compute_indicators`` →
[start, end] 절단 → 체결 시뮬레이션 → 성과지표. 산출물 3종을
``out_dir``(HITL 산출물 규약과 파일명 일치)에 저장한다.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from research_backtest.core.hitl import gates
from research_backtest.core.hitl.models import StrategyReview
from research_backtest.core.market.calendar import as_date
from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.data import (
    assert_no_lookahead,
    build_backtest_frame,
    truncate_to_window,
)
from research_backtest.quant.backtest.engine import EngineResult, run_backtest
from research_backtest.quant.backtest.metrics import BacktestResult, compute_backtest_metrics
from research_backtest.quant.strategy.compiler import compile_strategy, entry_signal, exit_signal
from research_backtest.quant.strategy.indicators import compute_indicators
from research_backtest.quant.strategy.schema import parse_strategy_spec

logger = logging.getLogger("r2b.backtest.runner")

# 산출물 파일명 (명세 A6 §5, README §2 Project 2 / 1804 HITL 규약과 일치)
BACKTEST_RESULT_FILENAME = "backtest_result.json"
TRADE_LOG_FILENAME = "trade_log.csv"
DAILY_PORTFOLIO_FILENAME = "daily_portfolio.csv"

# 벤치마크명 → 지수 종목코드 (A3 index_{code} 디렉토리, README §24)
BENCHMARK_INDEX_CODES = {"KOSPI": "1001", "KOSDAQ": "2001"}


def execute_approved_strategy(
    review: StrategyReview,
    *,
    data_dir: Path,
    stock_code: str,
    corp_code: str,
    start_date: date,
    end_date: date,
    out_dir: Path,
    backtest_config: BacktestConfig,
    fs_scope: str = "CFS",
) -> BacktestResult:
    """승인된 전략을 실데이터로 백테스트하고 산출물 3종을 저장한다 (명세 A6 §5).

    첫 줄에서 :func:`gates.ensure_strategy_approved`로 승인을 강제한다 —
    미승인 review는 :class:`ApprovalGateError`로 즉시 거부하고 산출물을 만들지
    않는다. ``review.final_strategy``를 A5로 재검증(``parse_strategy_spec`` →
    ``compile_strategy``)해 승인본이 DSL 규칙을 여전히 만족하는지 확인한다.
    """
    gates.ensure_strategy_approved(review)

    spec = parse_strategy_spec(review.final_strategy)
    compiled = compile_strategy(spec)

    daily = _load_daily(data_dir, stock_code)
    metrics = _load_metrics(data_dir, corp_code)

    # financial as-of join(워밍업 보존) → 룩어헤드 방어 검증
    joined = build_backtest_frame(
        daily, metrics, fs_scope=fs_scope, start_date=start_date, end_date=end_date
    )
    assert_no_lookahead(joined)

    # 지표 계산 → [start, end] 절단 (rolling 지표는 절단 전에 계산; 명세 A6 §2)
    with_indicators = compute_indicators(joined, compiled.required_columns)
    entry = entry_signal(compiled, with_indicators)
    exit_ = exit_signal(compiled, with_indicators)

    frame = truncate_to_window(with_indicators, start_date, end_date)
    entry = entry.reindex(frame.index)
    exit_ = exit_.reindex(frame.index)

    engine_result = run_backtest(frame, entry, exit_, compiled.position_rules, backtest_config)

    benchmark_close = _load_benchmark_close(data_dir, backtest_config.benchmark, frame.index)
    result = compute_backtest_metrics(
        engine_result=engine_result,
        asset_frame=frame,
        benchmark_close=benchmark_close,
        config=backtest_config,
        strategy_name=spec.strategy_name,
        start_date=start_date,
        end_date=end_date,
        fs_scope=fs_scope,
    )

    _save_artifacts(out_dir, result, engine_result)
    logger.info(
        "백테스트 완료 strategy=%s [%s~%s] 거래=%d 누적수익률=%s",
        spec.strategy_name,
        start_date,
        end_date,
        result.num_trades,
        result.cumulative_return,
    )
    return result


# --- 데이터 로드 -------------------------------------------------------------


def _load_daily(data_dir: Path, stock_code: str) -> pd.DataFrame:
    """A3 정규화 종목 daily.parquet을 로드한다 (open/high/low/close/volume/수급)."""
    path = data_dir / "normalized" / "market" / stock_code / "daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"종목 daily.parquet이 없습니다: {path} (r2b collect-market으로 먼저 수집)"
        )
    return pd.read_parquet(path)


def _load_metrics(data_dir: Path, corp_code: str) -> pd.DataFrame:
    """A4 financial_metrics.parquet을 로드한다 (available_from 포함)."""
    path = data_dir / "normalized" / "financials" / corp_code / "financial_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"financial_metrics.parquet이 없습니다: {path} (r2b build-financials로 먼저 생성)"
        )
    return pd.read_parquet(path)


def _load_benchmark_close(data_dir: Path, benchmark: str, index: pd.Index) -> pd.Series:
    """벤치마크 지수 종가를 대상 거래일(``index``)에 정렬해 반환한다.

    지수 데이터가 없으면(부분 수집 모드) 전 구간 NaN Series로 대체한다 —
    metrics가 벤치마크 지표를 None으로 처리한다(명세 A6 §4).
    """
    index_code = BENCHMARK_INDEX_CODES.get(benchmark, benchmark)
    path = data_dir / "normalized" / "market" / f"index_{index_code}" / "daily.parquet"
    if not path.exists():
        logger.warning("벤치마크 지수 데이터 없음: %s — 벤치마크 지표는 None 처리", path)
        return pd.Series([float("nan")] * len(index), index=index, name="benchmark_close")
    frame = pd.read_parquet(path)
    dates = [as_date(value) for value in frame["date"]]
    close = pd.Series(frame["close"].to_numpy(dtype="float64"), index=pd.Index(dates, name="date"))
    reindexed: pd.Series = close.reindex(index)
    reindexed.name = "benchmark_close"
    return reindexed


# --- 산출물 저장 -------------------------------------------------------------


def _save_artifacts(out_dir: Path, result: BacktestResult, engine_result: EngineResult) -> None:
    """backtest_result.json · trade_log.csv · daily_portfolio.csv 저장 (명세 A6 §5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BACKTEST_RESULT_FILENAME).write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    engine_result.trade_frame().to_csv(out_dir / TRADE_LOG_FILENAME, index=False)
    engine_result.daily_frame().to_csv(out_dir / DAILY_PORTFOLIO_FILENAME, index=False)
