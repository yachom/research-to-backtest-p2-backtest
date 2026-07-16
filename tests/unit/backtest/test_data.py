"""data.py — as-of join·assert_no_lookahead·절단 테스트 (명세 A6 §2·§6, README §22·§28.3).

§28.3-1(공시일 이전 가격 행에 재무값이 존재하면 실패)을 포함한다.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date

import pandas as pd
import pytest

from research_backtest.core.exceptions import DataValidationError, LookaheadError
from research_backtest.quant.backtest.data import (
    FINANCIAL_COLUMNS_ATTR,
    METRIC_AVAILABLE_FROM_ATTR,
    assert_no_lookahead,
    build_backtest_frame,
    truncate_to_window,
)

DailyFactory = Callable[..., pd.DataFrame]
MetricsFactory = Callable[..., pd.DataFrame]

FROM = date(2024, 1, 1)
YOY = "operating_income_yoy"


def _daily5(make_daily: DailyFactory) -> pd.DataFrame:
    """2024-01-01(월)~01-05(금) 5영업일 daily(종가=진입가 무관 상수)."""
    return make_daily(opens=[100] * 5, closes=[100] * 5, start=FROM)


# --- as-of join 기본 (명세 A6 §2, README §22.2) ------------------------------


def test_asof_join_nan_before_value_after_and_boundary(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """공시 전 NaN · available_from 당일부터 값 노출(경계일 포함)."""
    daily = _daily5(make_daily)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    values = frame[YOY].tolist()
    assert math.isnan(values[0]) and math.isnan(values[1])  # 01-01, 01-02: 공시 전
    assert values[2] == 5.0  # 01-03: available_from 당일 = 노출
    assert values[3] == 5.0 and values[4] == 5.0  # 다음 공시 전까지 유지(§22.2)


def test_asof_join_replaces_at_next_disclosure(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """다음 공시 available_from부터 값이 교체된다."""
    daily = _daily5(make_daily)
    metrics = make_metrics(
        [
            (YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2)),
            (YOY, "CFS", date(2024, 1, 4), 7.0, date(2024, 1, 3)),
        ]
    )
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    assert frame[YOY].tolist()[2:] == [5.0, 7.0, 7.0]


def test_scope_filter_ignores_other_scope(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """fs_scope 필터 — 다른 scope 값은 병합하지 않는다."""
    daily = _daily5(make_daily)
    metrics = make_metrics(
        [
            (YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2)),
            (YOY, "OFS", date(2024, 1, 3), 99.0, date(2024, 1, 2)),
        ]
    )
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    assert frame[YOY].dropna().unique().tolist() == [5.0]  # OFS 99.0은 미포함


def test_duplicate_available_from_prefers_latest_rcept(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """동일 available_from 중복은 rcept_dt 최신을 채택한다(정정 반영, 명세 A6 §2)."""
    daily = _daily5(make_daily)
    metrics = make_metrics(
        [
            (YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2)),  # 원본
            (YOY, "CFS", date(2024, 1, 3), 6.0, date(2024, 1, 3)),  # 정정(rcept 최신)
        ]
    )
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    assert frame[YOY].tolist()[2:] == [6.0, 6.0, 6.0]  # 정정값 6.0


# --- §28.3-1: 공시일 이전 행에 재무값이 존재하면 실패 -------------------------


def test_no_financial_value_before_available_from(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """§28.3-1: 공시(available_from) 이전 거래일에는 재무값이 절대 없다."""
    daily = _daily5(make_daily)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 4), 5.0, date(2024, 1, 3))])
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    before = frame.loc[[d for d in frame.index if d < date(2024, 1, 4)], YOY]
    assert before.isna().all()  # 01-01~01-03: 재무값 없음
    assert_no_lookahead(frame)  # 올바른 join은 방어 검증 통과


def test_assert_no_lookahead_raises_on_future_value() -> None:
    """§28.3-1(방어선): 값이 available_from 이전 거래일에 노출되면 LookaheadError."""
    idx = pd.Index([date(2024, 1, 2), date(2024, 1, 3)], name="date")
    frame = pd.DataFrame({"close": [100.0, 101.0], YOY: [5.0, 5.0]}, index=idx)
    # 01-02 행에 available_from 01-10 값이 노출된 위반 상태를 인위적으로 구성
    frame.attrs[METRIC_AVAILABLE_FROM_ATTR] = pd.DataFrame(
        {YOY: [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-10")]}, index=idx
    )
    frame.attrs[FINANCIAL_COLUMNS_ATTR] = [YOY]
    with pytest.raises(LookaheadError, match="룩어헤드 위반"):
        assert_no_lookahead(frame)


def test_assert_no_lookahead_requires_meta() -> None:
    """메타 없는 프레임(잘못된 경로 생성)은 LookaheadError로 거부한다."""
    frame = pd.DataFrame({"close": [1.0]}, index=pd.Index([date(2024, 1, 2)], name="date"))
    with pytest.raises(LookaheadError, match="메타가 없습니다"):
        assert_no_lookahead(frame)


# --- 워밍업 보존·절단 (명세 A6 §2) -------------------------------------------


def test_build_preserves_warmup_and_truncates_end(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """start 이전 워밍업은 남기고 end 이후는 버린다(지표 계산 뒤 절단 위함)."""
    daily = make_daily(opens=[100] * 10, closes=[100] * 10, start=FROM)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    frame = build_backtest_frame(
        daily,
        metrics,
        fs_scope="CFS",
        start_date=date(2024, 1, 8),  # 워밍업: 01-01~01-05는 start 이전
        end_date=date(2024, 1, 10),
    )
    assert frame.index[0] == date(2024, 1, 1)  # 워밍업 보존
    assert frame.index[-1] <= date(2024, 1, 10)  # end 이후 없음


def test_truncate_to_window_slices_and_keeps_meta(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """truncate_to_window가 [start,end]로 자르고 룩어헤드 메타를 이어받는다."""
    daily = make_daily(opens=[100] * 10, closes=[100] * 10, start=FROM)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    frame = build_backtest_frame(
        daily, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 12)
    )
    windowed = truncate_to_window(frame, date(2024, 1, 4), date(2024, 1, 9))
    assert windowed.index[0] == date(2024, 1, 4)
    assert windowed.index[-1] == date(2024, 1, 9)
    assert_no_lookahead(windowed)  # 절단 후에도 메타로 검증 가능


def test_build_rejects_out_of_range_window(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """daily와 겹치지 않는 구간은 DataValidationError."""
    daily = _daily5(make_daily)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    with pytest.raises(DataValidationError):
        build_backtest_frame(
            daily, metrics, fs_scope="CFS", start_date=date(2025, 1, 1), end_date=date(2025, 1, 5)
        )


def test_build_rejects_start_after_end(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    daily = _daily5(make_daily)
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    with pytest.raises(ValueError, match="start_date"):
        build_backtest_frame(
            daily, metrics, fs_scope="CFS", start_date=date(2024, 1, 5), end_date=FROM
        )


def test_build_accepts_date_column_or_index(
    make_daily: DailyFactory, make_metrics: MetricsFactory
) -> None:
    """daily가 date 컬럼(정규화 산출)이든 index든 동일하게 처리한다."""
    daily_idx = _daily5(make_daily)
    daily_col = daily_idx.reset_index()  # date를 컬럼으로
    metrics = make_metrics([(YOY, "CFS", date(2024, 1, 3), 5.0, date(2024, 1, 2))])
    f_idx = build_backtest_frame(
        daily_idx, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    f_col = build_backtest_frame(
        daily_col, metrics, fs_scope="CFS", start_date=FROM, end_date=date(2024, 1, 5)
    )
    assert f_idx[YOY].fillna(-1).tolist() == f_col[YOY].fillna(-1).tolist()
