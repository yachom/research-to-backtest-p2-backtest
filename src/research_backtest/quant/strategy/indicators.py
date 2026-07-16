"""가격·수급 지표 계산 — no-lookahead (README §21.2~§21.3, §22.1, 명세 A5 §4).

입력은 A3 daily.parquet 스키마의 DataFrame이다: ``date``(오름차순 index 또는
정렬된 컬럼), ``open, high, low, close, volume, foreign_net_buy_value,
institution_net_buy_value``. FINANCIAL 지표(README §21.1)는 여기서
계산하지 않는다 — A4가 만든 ``financial_metrics``를 A6가 as-of join으로
공급한다(README §22.2). 이미 프레임에 존재하는 컬럼(예: A6가 재무 지표를
미리 join해 둔 경우)은 재계산하지 않고 그대로 둔다 — lag는 소스와 무관하게
"이미 프레임에 있는 어떤 컬럼이든" 적용 가능한 범용 연산이다.

**no-lookahead 원칙(README §22.1 취지)**: 모든 지표는 rolling/ewm 같은
인과적(causal) 연산만 사용한다 — t행의 값은 0..t행의 데이터만으로
결정되며, t+1행 이후 데이터가 바뀌어도 t행 값은 변하지 않는다(프레임을
t까지 잘라도 t행 값이 동일해야 한다 — tests/unit/strategy의 property
테스트로 검증). 워밍업 구간(rolling/ewm 최소 관측치 미충족)은 NaN으로
남기고 채우지 않는다 — 컴파일러가 NaN 관여 비교를 False로 처리한다
(명세 A5 §5).
"""

from __future__ import annotations

import math
import re

import pandas as pd

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.registry import IndicatorSource, resolve_indicator

_SMA_RE = re.compile(r"^sma_(\d+)$")
_ROLLING_HIGH_RE = re.compile(r"^rolling_high_(\d+)$")
_RETURN_RE = re.compile(r"^return_(\d+)d$")

_PASSTHROUGH_COLUMNS = frozenset({"close", "open", "high", "low", "volume"})

_WILDER_PERIOD = 14
_WILDER_ALPHA = 1 / _WILDER_PERIOD
_TRADING_DAYS_PER_YEAR = 252


def compute_indicators(daily: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    """요구된 지표 컬럼을 추가해 반환한다 (명세 A5 §4).

    ``required``의 각 이름은 :func:`registry.resolve_indicator`로 먼저 전부
    해석한다 — 미지원 지표명은 여기서 :class:`StrategyValidationError`로
    거부된다. FINANCIAL 지표는 계산하지 않는다(A6가 as-of join으로 공급).
    lag(``_lag{n}``)는 base 컬럼이 계산된 "이후" 프레임에 이미 존재하는
    경우에만 적용할 수 있다 — FINANCIAL 지표에 lag를 적용하려면 A6가 join을
    먼저 수행한 프레임을 넘겨야 한다(그렇지 않으면 명시적으로 실패한다).

    반환된 프레임은 ``daily``의 사본이며 원본을 변경하지 않는다.
    """
    resolved = [resolve_indicator(name) for name in sorted(required)]

    frame = daily.copy()

    # 1단계: PRICE·FLOW base 컬럼을 계산한다 (FINANCIAL은 건드리지 않는다).
    for info in resolved:
        if info.source is IndicatorSource.FINANCIAL:
            continue
        _ensure_base_column(frame, info.base)

    # 2단계: lag를 적용한다 — base가 1단계에서 계산됐거나(PRICE·FLOW) 호출자가
    # 미리 join해 둔 컬럼(예: FINANCIAL)이면 소스와 무관하게 동작한다.
    for info in resolved:
        if info.lag == 0:
            continue
        if info.base not in frame.columns:
            raise StrategyValidationError(
                f"지표 {info.name!r}의 기준 컬럼 {info.base!r}이 프레임에 없습니다. "
                "FINANCIAL 지표에 lag를 적용하려면 A6가 as-of join을 compute_indicators "
                "호출 이전에 수행해야 합니다(명세 A5 §4)."
            )
        frame[info.name] = frame[info.base].shift(info.lag)

    return frame


def _ensure_base_column(frame: pd.DataFrame, base: str) -> None:
    """PRICE·FLOW base 지표 1개를 계산해 ``frame``에 in-place로 추가한다.

    이미 존재하면(호출자가 미리 채워 둔 경우 포함) 재계산하지 않는다.
    """
    if base in frame.columns:
        return

    if base in _PASSTHROUGH_COLUMNS:
        raise StrategyValidationError(
            f"daily 프레임에 {base!r} 컬럼이 없습니다 (A3 daily.parquet 스키마 필요, 명세 A5 §4)."
        )

    if (match := _SMA_RE.match(base)) is not None:
        n = int(match.group(1))
        frame[base] = _column(frame, "close").rolling(n, min_periods=n).mean()
        return
    if (match := _ROLLING_HIGH_RE.match(base)) is not None:
        n = int(match.group(1))
        # 당일 고가 포함 — 돌파 판정용 직전 N일 고점은 호출자가 lag(_lag1)로 얻는다(README §23.4).
        frame[base] = _column(frame, "high").rolling(n, min_periods=n).max()
        return
    if (match := _RETURN_RE.match(base)) is not None:
        n = int(match.group(1))
        frame[base] = _column(frame, "close").pct_change(n, fill_method=None)
        return
    if base == "volatility_20":
        daily_return = _column(frame, "close").pct_change(fill_method=None)
        frame[base] = daily_return.rolling(20, min_periods=20).std() * math.sqrt(
            _TRADING_DAYS_PER_YEAR
        )
        return
    if base == "rsi_14":
        frame[base] = _wilder_rsi(_column(frame, "close"))
        return
    if base == "atr_14":
        frame[base] = _wilder_atr(
            _column(frame, "high"), _column(frame, "low"), _column(frame, "close")
        )
        return
    if base == "foreign_net_buy_5d":
        frame[base] = _column(frame, "foreign_net_buy_value").rolling(5, min_periods=5).sum()
        return
    if base == "foreign_net_buy_20d":
        frame[base] = _column(frame, "foreign_net_buy_value").rolling(20, min_periods=20).sum()
        return
    if base == "institution_net_buy_5d":
        frame[base] = _column(frame, "institution_net_buy_value").rolling(5, min_periods=5).sum()
        return
    if base == "institution_net_buy_20d":
        frame[base] = _column(frame, "institution_net_buy_value").rolling(20, min_periods=20).sum()
        return

    # 화이트리스트(registry)에는 있으나 계산 정의가 없는 경우 — 레지스트리·계산기 불일치(버그).
    raise StrategyValidationError(
        f"지표 {base!r}에 대한 계산 정의가 없습니다(레지스트리·계산기 불일치)."
    )


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """daily 스키마 컬럼(예: close, foreign_net_buy_value)을 조회한다 — 없으면 명시적으로 실패."""
    if name not in frame.columns:
        raise StrategyValidationError(
            f"daily 프레임에 {name!r} 컬럼이 없습니다 (A3 daily.parquet 스키마 필요, 명세 A5 §4)."
        )
    series: pd.Series = frame[name]
    return series


def _wilder_rsi(close: pd.Series) -> pd.Series:
    """Wilder RSI(14) — README §21.2 ``rsi_14``.

    ``delta = close.diff()``; ``gain = max(delta, 0)``, ``loss = max(-delta, 0)``.
    ``avg_gain``·``avg_loss``는 Wilder 지수평활(EMA, ``alpha=1/14``,
    ``adjust=False``) — 14개 미만 관측 구간은 NaN(``min_periods=14``).
    ``RS = avg_gain / avg_loss``; ``RSI = 100 - 100 / (1 + RS)``.
    ``avg_loss == 0``이면 RSI는 100에 수렴하고, 등락이 전혀 없어
    ``avg_gain == avg_loss == 0``이면 0/0으로 NaN이다(횡보 구간, 정의역 밖).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=_WILDER_ALPHA, adjust=False, min_periods=_WILDER_PERIOD).mean()
    avg_loss = loss.ewm(alpha=_WILDER_ALPHA, adjust=False, min_periods=_WILDER_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi: pd.Series = 100 - (100 / (1 + rs))
    return rsi


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder ATR(14) — README §21.2 ``atr_14``.

    ``TR[t] = max(high[t]-low[t], |high[t]-close[t-1]|, |low[t]-close[t-1]|)``
    (첫 행은 이전 종가가 없으므로 ``high-low``만 사용). ATR은 TR의 Wilder
    지수평활(EMA, ``alpha=1/14``, ``adjust=False``, ``min_periods=14``)이다.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr: pd.Series = true_range.ewm(
        alpha=_WILDER_ALPHA, adjust=False, min_periods=_WILDER_PERIOD
    ).mean()
    return atr
