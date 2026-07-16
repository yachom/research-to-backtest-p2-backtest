"""백테스트 엔진(quant.backtest) 단위 테스트 공용 픽스처 (명세 A6 §6) — 전부 오프라인.

손으로 만든 소형 시계열로 as-of join·체결·지표를 손계산 대조한다. 실데이터·
네트워크는 쓰지 않는다(integration은 tests/integration/test_backtest_run.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta

import pandas as pd
import pytest

from research_backtest.core.hitl.models import StrategyReview
from research_backtest.quant.backtest.costs import BacktestConfig

DailyFactory = Callable[..., pd.DataFrame]
MetricsFactory = Callable[..., pd.DataFrame]
SignalFactory = Callable[[pd.Index, Sequence[date]], pd.Series]
ConfigFactory = Callable[..., BacktestConfig]


def _weekday_dates(start: date, n: int) -> list[date]:
    """주말을 건너뛴 연속 영업일 n개(단위 테스트용 — 공휴일 무시)."""
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


@pytest.fixture
def make_daily() -> DailyFactory:
    """open/close(및 옵션 high/low)로 date-index daily 프레임을 만든다.

    high/low 미지정 시 max(open,close)/min(open,close)로 채운다. index는
    datetime.date 오름차순(주말 제외 연속 영업일)이다.
    """

    def _make(
        *,
        opens: Sequence[float],
        closes: Sequence[float],
        highs: Sequence[float] | None = None,
        lows: Sequence[float] | None = None,
        foreign_net_buy_value: Sequence[float] | None = None,
        institution_net_buy_value: Sequence[float] | None = None,
        start: date = date(2024, 1, 1),
    ) -> pd.DataFrame:
        n = len(opens)
        assert len(closes) == n
        dates = _weekday_dates(start, n)
        pairs = list(zip(opens, closes, strict=True))
        high_vals = highs if highs is not None else [max(o, c) for o, c in pairs]
        low_vals = lows if lows is not None else [min(o, c) for o, c in pairs]
        frame = pd.DataFrame(
            {
                "open": list(opens),
                "high": list(high_vals),
                "low": list(low_vals),
                "close": list(closes),
                "volume": [1000] * n,
                "foreign_net_buy_value": (
                    list(foreign_net_buy_value) if foreign_net_buy_value is not None else [0] * n
                ),
                "institution_net_buy_value": (
                    list(institution_net_buy_value)
                    if institution_net_buy_value is not None
                    else [0] * n
                ),
            },
            index=pd.Index(dates, name="date"),
        )
        return frame

    return _make


@pytest.fixture
def make_metrics() -> MetricsFactory:
    """A4 financial_metrics 스키마의 소형 long 프레임을 만든다.

    각 원소는 (metric_id, fs_scope, available_from, value, rcept_dt) 튜플이다.
    """

    def _make(rows: Sequence[tuple[str, str, date, float, date | None]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric_id": [r[0] for r in rows],
                "fs_scope": [r[1] for r in rows],
                "available_from": [r[2] for r in rows],
                "value": [r[3] for r in rows],
                "rcept_dt": [r[4] for r in rows],
            }
        )

    return _make


@pytest.fixture
def make_signal() -> SignalFactory:
    """지정한 날짜에만 True인 bool Series(index 정렬)를 만든다."""

    def _make(index: pd.Index, true_dates: Sequence[date]) -> pd.Series:
        return pd.Series([d in set(true_dates) for d in index], index=index)

    return _make


@pytest.fixture
def bt_config() -> ConfigFactory:
    """비용 0·초기자본 100만원 기본 BacktestConfig 팩토리(비용은 테스트에서 주입)."""

    def _make(**overrides: object) -> BacktestConfig:
        params: dict[str, object] = {
            "commission_rate": 0.0,
            "sell_tax_rate": 0.0,
            "slippage_rate": 0.0,
            "initial_cash": 1_000_000.0,
        }
        params.update(overrides)
        return BacktestConfig.model_validate(params)

    return _make


@pytest.fixture
def approved_review() -> Callable[[dict[str, object]], StrategyReview]:
    """승인된 StrategyReview를 만드는 팩토리(final_strategy만 주입)."""

    def _make(strategy: dict[str, object]) -> StrategyReview:
        return StrategyReview(
            review_id="rv-test",
            hypothesis_id="hyp-test",
            llm_draft_strategy=strategy,
            final_strategy=strategy,
            modifications=[],
            approval_reason="단위 테스트 승인",
            approved_by="tester",
            approved_at="2026-07-14T09:00:00+09:00",
        )

    return _make
