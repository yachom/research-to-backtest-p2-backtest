"""전략 DSL 지표 레지스트리 — 허용 지표 화이트리스트 + lag 문법 (README §21,
명세 A5 §3, MILESTONES §5 정오표 1 해소).

지표명은 세 계열로 분류된다(:class:`IndicatorSource`) — A6는 이 분류로
데이터 준비 방식을 결정한다: PRICE·FLOW는 :mod:`indicators`가 A3
daily.parquet에서 계산하고, FINANCIAL은 A4 ``financial_metrics``를
as-of join한 컬럼을 그대로 참조한다(여기서는 계산하지 않는다).

**lag 문법**: 임의의 화이트리스트 지표 ``base``에 대해 ``{base}_lag{n}``
(n>=1 정수)는 ``base``를 ``n``일 지연시킨 값을 뜻한다. README §23.4가
쓰는 ``rolling_high_60_lag1``이 이 문법의 근거다 — README §21.2 목록에는
``rolling_high_60``만 있고 lagged 변형이 정식 등록되어 있지 않던 불일치를
(MILESTONES 정오표 1) 이 문법으로 해소한다.

미지원 지표명은 항상 :class:`StrategyValidationError`로 거부하고, 지표명과
허용 목록 힌트를 메시지에 담는다(README §31 M8 "지원하지 않는 변수 처리") —
조용히 무시하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from research_backtest.core.exceptions import StrategyValidationError


class IndicatorSource(StrEnum):
    """지표의 데이터 원천 분류 (명세 A5 §3) — A6가 데이터 준비(조인)에 사용한다."""

    PRICE = "PRICE"  # ohlcv에서 계산 (quant.strategy.indicators)
    FLOW = "FLOW"  # 투자자 수급에서 계산 (quant.strategy.indicators)
    FINANCIAL = "FINANCIAL"  # A4 financial_metrics 컬럼 참조 — as-of join, 여기서 계산 안 함


# README §21.1 허용 재무지표 — A4 metric_id와 동일 명명(명세 A4, "A5 DSL이 같은
# 이름을 컬럼으로 참조") — 여기서는 컬럼 참조만 하고 값 계산은 A4 책임이다.
FINANCIAL_INDICATORS: frozenset[str] = frozenset(
    {
        "revenue_yoy",
        "operating_income_yoy",
        "net_income_yoy",
        "operating_margin",
        "roe",
        "roa",
        "debt_ratio",
        "net_debt",
        "operating_cash_flow",
        "free_cash_flow",
        "inventory_yoy",
        "receivables_yoy",
    }
)

# README §21.2 허용 가격지표
PRICE_INDICATORS: frozenset[str] = frozenset(
    {
        "close",
        "open",
        "high",
        "low",
        "volume",
        "sma_5",
        "sma_20",
        "sma_60",
        "sma_120",
        "rolling_high_20",
        "rolling_high_60",
        "return_20d",
        "return_60d",
        "volatility_20",
        "rsi_14",
        "atr_14",
    }
)

# README §21.3 허용 수급지표
FLOW_INDICATORS: frozenset[str] = frozenset(
    {
        "foreign_net_buy_5d",
        "foreign_net_buy_20d",
        "institution_net_buy_5d",
        "institution_net_buy_20d",
    }
)

_SOURCE_BY_INDICATOR: dict[str, IndicatorSource] = {
    **dict.fromkeys(FINANCIAL_INDICATORS, IndicatorSource.FINANCIAL),
    **dict.fromkeys(PRICE_INDICATORS, IndicatorSource.PRICE),
    **dict.fromkeys(FLOW_INDICATORS, IndicatorSource.FLOW),
}

ALL_BASE_INDICATORS: frozenset[str] = frozenset(_SOURCE_BY_INDICATOR)

# {base}_lag{n} — n>=1 정수 (명세 A5 §3). base 자체가 "_lagN"으로 끝나는 화이트리스트
# 지표는 없으므로 탐욕적 매칭이 잘못된 base를 골라낼 위험은 없다.
_LAG_SUFFIX_RE = re.compile(r"^(?P<base>.+)_lag(?P<n>\d+)$")


@dataclass(frozen=True)
class ResolvedIndicator:
    """지표명 1개의 해석 결과 (명세 A5 §3)."""

    name: str  # 원본 지표명 그대로(lag 포함) — 프레임 컬럼명과 동일
    base: str  # lag 접미사를 제거한 base 지표명 — 화이트리스트 원소
    lag: int  # 0이면 lag 없음, n(>=1)이면 base를 n일 지연(.shift(n))
    source: IndicatorSource


def resolve_indicator(name: str) -> ResolvedIndicator:
    """지표명을 화이트리스트·lag 문법으로 해석한다 (README §21, 명세 A5 §3).

    ``name``이 화이트리스트에 직접 있으면 lag 없음으로 해석한다. 그렇지
    않으면 ``{base}_lag{n}`` 형태(n>=1)인지 확인해 base가 화이트리스트에
    있으면 유효로 처리한다. 둘 다 아니면 :class:`StrategyValidationError`에
    지표명과 허용 목록을 담아 던진다.
    """
    source = _SOURCE_BY_INDICATOR.get(name)
    if source is not None:
        return ResolvedIndicator(name=name, base=name, lag=0, source=source)

    match = _LAG_SUFFIX_RE.match(name)
    if match is not None:
        base = match.group("base")
        lag = int(match.group("n"))
        base_source = _SOURCE_BY_INDICATOR.get(base)
        if lag >= 1 and base_source is not None:
            return ResolvedIndicator(name=name, base=base, lag=lag, source=base_source)

    raise StrategyValidationError(
        f"지원하지 않는 지표입니다: {name!r}. "
        f"허용 지표: {sorted(ALL_BASE_INDICATORS)} — "
        "lag 문법 '{base}_lag{n}'(n>=1)도 가능합니다."
    )
