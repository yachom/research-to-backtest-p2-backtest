"""데이터 준비 — daily(A3) + financial_metrics(A4) as-of join (명세 A6 §2, README §22).

**가장 중요한 정확성 요구(README §22.1)**: 재무 수치는 회계기간 종료일이 아니라
공시 이용 가능일(``available_from``)부터만 보여야 한다. 이 모듈은 metric을
``available_from`` 기준 ``merge_asof``로 daily에 병합해, 거래일 t에는
``available_from <= t``인 가장 최근 값만 노출한다(다음 공시 전까지 직전 값 유지,
§22.2). period_end 기준 병합·최신값 소급 적용은 금지다(§22.3) — join 키는 오직
``available_from``이며, :func:`assert_no_lookahead`가 병합 결과를 방어 검증한다.

**호출 순서(A5 인계 권장, 명세 A6 §2)**: financial join → ``compute_indicators``
→ 절단. rolling 지표가 워밍업 구간을 보려면 절단이 지표 계산 뒤에 와야 하므로,
:func:`build_backtest_frame`은 start_date 이전 워밍업 행을 **남기고**(끝만
end_date로 절단), 최종 [start, end] 절단은 :func:`truncate_to_window`가
``compute_indicators`` 이후에 수행한다. build_backtest_frame은 지표 계산에 필요한
compiled 전략을 받지 않으므로 여기서 지표를 계산하지 않는다.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from research_backtest.core.exceptions import DataValidationError, LookaheadError
from research_backtest.core.market.calendar import as_date

logger = logging.getLogger("r2b.backtest.data")

# 병합 후 frame.attrs에 붙이는 메타 키 (assert_no_lookahead·truncate_to_window가 소비)
FINANCIAL_COLUMNS_ATTR = "financial_columns"
METRIC_AVAILABLE_FROM_ATTR = "metric_available_from"
BACKTEST_START_ATTR = "backtest_start"
BACKTEST_END_ATTR = "backtest_end"
FS_SCOPE_ATTR = "fs_scope"

_OHLCV_REQUIRED = ("open", "high", "low", "close")


def build_backtest_frame(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    fs_scope: str = "CFS",
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """daily에 financial_metrics를 available_from 기준 as-of join한다 (명세 A6 §2).

    - ``metrics``를 ``fs_scope``로 필터 → metric_id별 wide 컬럼(컬럼명 =
      metric_id, A5 financial_columns와 동일 명명)으로 ``merge_asof``(방향
      backward)로 병합한다. 거래일 t에는 ``available_from <= t``인 가장 최근
      metric 값만 보이고, 다음 공시 전까지 직전 값이 유지된다(§22.2).
    - 동일 metric의 ``available_from`` 중복(같은 날 두 값 — 이론상 정정)은
      ``rcept_dt`` 최신을 우선하고 발생 사실을 로그로 남긴다(명세 A6 §2).
    - **워밍업 보존**: 결과는 daily 시작부터 ``end_date``까지의 행을 담는다
      (start_date 이전 행을 남긴다). 최종 [start, end] 절단은
      :func:`truncate_to_window`가 ``compute_indicators`` 이후 수행한다.
    - ``frame.attrs``에 룩어헤드 방어 검증용 메타(각 행의 metric 값이 유래한
      available_from)를 붙인다 — :func:`assert_no_lookahead`가 소비한다.

    반환 프레임은 ``date``(datetime.date) 오름차순 index를 가지며 원본을
    변경하지 않는다.
    """
    if start_date > end_date:
        raise ValueError(f"start_date({start_date})가 end_date({end_date})보다 큽니다.")

    work = _prepare_daily(daily)
    # 끝만 절단(미래 미사용) — start 이전 워밍업은 남긴다(명세 A6 §2)
    work = work.loc[work["date"] <= end_date].reset_index(drop=True)
    if work.empty:
        raise DataValidationError(
            f"end_date({end_date}) 이전 거래일 데이터가 없습니다 — daily 범위를 확인하세요."
        )
    if work["date"].iloc[-1] < start_date:
        raise DataValidationError(
            f"[{start_date}, {end_date}] 구간과 겹치는 거래일이 없습니다 "
            f"(daily 최종일 {work['date'].iloc[-1]})."
        )

    work["_date"] = pd.to_datetime(work["date"])
    scope_metrics = metrics.loc[metrics["fs_scope"] == fs_scope]
    metric_ids = sorted(str(m) for m in scope_metrics["metric_id"].dropna().unique())

    meta = pd.DataFrame(index=work.index)
    for metric_id in metric_ids:
        values, source_af = _asof_one_metric(work, scope_metrics, metric_id)
        work[metric_id] = values
        meta[metric_id] = source_af

    frame = work.drop(columns=["_date"]).set_index("date")
    frame.index.name = "date"
    meta.index = frame.index

    frame.attrs[FINANCIAL_COLUMNS_ATTR] = list(metric_ids)
    frame.attrs[METRIC_AVAILABLE_FROM_ATTR] = meta
    frame.attrs[BACKTEST_START_ATTR] = start_date
    frame.attrs[BACKTEST_END_ATTR] = end_date
    frame.attrs[FS_SCOPE_ATTR] = fs_scope
    return frame


def assert_no_lookahead(frame: pd.DataFrame) -> None:
    """병합 결과의 룩어헤드 방어 검증 (명세 A6 §2, README §22.3).

    임의 거래일 t의 각 financial 컬럼 값이 실제로 ``available_from <= t``인
    공시에서 유래했는지 확인한다 — 위반 시 :class:`LookaheadError`.
    :func:`build_backtest_frame`이 ``frame.attrs``에 심어 둔 per-cell
    available_from 메타로 검증하며, 메타가 없으면(잘못된 경로로 만든 프레임)
    역시 :class:`LookaheadError`로 거부한다.

    ``compute_indicators``·:func:`truncate_to_window` 이후에 호출해도 되도록
    메타를 현재 frame.index에 맞춰 재정렬한다.
    """
    meta = frame.attrs.get(METRIC_AVAILABLE_FROM_ATTR)
    financial_columns = frame.attrs.get(FINANCIAL_COLUMNS_ATTR, [])
    if meta is None:
        raise LookaheadError(
            "룩어헤드 검증 메타가 없습니다 — build_backtest_frame으로 생성한 프레임이 아닙니다."
        )

    meta = meta.reindex(frame.index)
    row_dates = pd.to_datetime(pd.Series(frame.index, index=frame.index))

    violations: list[str] = []
    for column in financial_columns:
        if column not in frame.columns:
            continue
        present = frame[column].notna().to_numpy()
        source_af = pd.to_datetime(meta[column])
        # 값이 있는데 출처 available_from이 없거나(NaT) 거래일보다 미래면 위반
        bad_missing = present & source_af.isna().to_numpy()
        bad_future = (
            present & ~source_af.isna().to_numpy() & (source_af.to_numpy() > row_dates.to_numpy())
        )
        for raw_pos in np.flatnonzero(bad_missing | bad_future):
            pos = int(raw_pos)
            trade_day = frame.index[pos]
            af_value = meta[column].iloc[pos]
            violations.append(f"{column}@{trade_day}: 출처 available_from={af_value!r} > 거래일")

    if violations:
        raise LookaheadError(
            "as-of join 룩어헤드 위반(재무값이 공시 이전 거래일에 노출됨): "
            + "; ".join(violations[:10])
            + (" …" if len(violations) > 10 else "")
        )


def truncate_to_window(frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """지표 계산이 끝난 프레임을 백테스트 구간 [start, end]로 절단한다 (명세 A6 §2).

    ``build_backtest_frame`` → ``compute_indicators`` **이후** 호출한다 —
    rolling 지표는 절단 전에 계산돼 있어야 워밍업이 반영된다. ``frame.attrs``
    (룩어헤드 메타 포함)를 이어받고 메타도 같은 구간으로 슬라이스한다.
    """
    index_dates = list(frame.index)
    mask = np.array([start_date <= d <= end_date for d in index_dates], dtype=bool)
    out = frame.loc[mask].copy()
    out.attrs = dict(frame.attrs)
    meta = frame.attrs.get(METRIC_AVAILABLE_FROM_ATTR)
    if meta is not None:
        out.attrs[METRIC_AVAILABLE_FROM_ATTR] = meta.loc[mask]
    return out


# --- 내부 구현 ---------------------------------------------------------------


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """A3 daily를 date 컬럼(오름차순) + RangeIndex로 정규화한다 (명세 A3 daily 스키마).

    date는 컬럼(정규화 산출)·index(테스트 편의) 어느 쪽이든 받아 datetime.date로
    통일한다. OHLCV 필수 컬럼 부재·date 중복은 :class:`DataValidationError`.
    """
    frame = daily.copy()
    if "date" in frame.columns:
        dates = [as_date(value) for value in frame["date"]]
        frame = frame.drop(columns=["date"])
    else:
        dates = [as_date(value) for value in frame.index]
    frame = frame.reset_index(drop=True)
    frame.insert(0, "date", dates)

    missing = [column for column in _OHLCV_REQUIRED if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"daily에 필수 OHLCV 컬럼이 없습니다: {missing} (A3 daily.parquet 스키마 필요)."
        )

    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        dup = frame.loc[frame["date"].duplicated(), "date"].tolist()
        raise DataValidationError(f"daily에 중복 거래일이 있습니다: {dup[:5]}")
    return frame


def _asof_one_metric(
    work: pd.DataFrame, scope_metrics: pd.DataFrame, metric_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """metric 1개를 available_from 기준으로 work에 as-of 병합한다.

    반환은 (값 배열, 출처 available_from datetime64 배열) — work 행과 위치가
    일치한다. 값이 없는 행(공시 이전)은 NaN·NaT다.
    """
    sub = scope_metrics.loc[
        scope_metrics["metric_id"] == metric_id, ["available_from", "value", "rcept_dt"]
    ].copy()
    sub["available_from"] = [as_date(value) for value in sub["available_from"]]
    sub["rcept_dt"] = [_as_date_or_none(value) for value in sub["rcept_dt"]]

    # 동일 available_from 중복은 rcept_dt 최신 우선(정정 반영) — 발생 사실 기록(명세 A6 §2)
    sub = sub.sort_values(["available_from", "rcept_dt"], na_position="first", kind="stable")
    before = len(sub)
    sub = sub.drop_duplicates(subset="available_from", keep="last")
    dropped = before - len(sub)
    if dropped:
        logger.info(
            "metric=%s available_from 중복 %d건 — rcept_dt 최신 우선 채택(정정 반영, 명세 A6 §2)",
            metric_id,
            dropped,
        )

    sub["_af"] = pd.to_datetime(sub["available_from"])
    sub = sub.sort_values("_af", kind="stable").reset_index(drop=True)

    merged = pd.merge_asof(
        work[["_date"]],
        sub[["_af", "value"]],
        left_on="_date",
        right_on="_af",
        direction="backward",
    )
    values: np.ndarray = merged["value"].to_numpy(dtype="float64")
    source_af: np.ndarray = merged["_af"].to_numpy()
    return values, source_af


def _as_date_or_none(value: object) -> date | None:
    """rcept_dt처럼 None이 가능한 날짜 값을 date|None으로 흡수한다."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return None
    return as_date(value)
