"""지표 레지스트리 테스트 — 화이트리스트·lag 문법·소스 분류 (README §21, 명세 A5 §3·§6)."""

import pytest

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.registry import (
    ALL_BASE_INDICATORS,
    FINANCIAL_INDICATORS,
    FLOW_INDICATORS,
    PRICE_INDICATORS,
    IndicatorSource,
    resolve_indicator,
)


def test_whitelist_partition_covers_all_base_indicators() -> None:
    """README §21.1~§21.3 세 목록은 서로 겹치지 않고 ALL_BASE_INDICATORS를 정확히 채운다."""
    assert FINANCIAL_INDICATORS.isdisjoint(PRICE_INDICATORS)
    assert FINANCIAL_INDICATORS.isdisjoint(FLOW_INDICATORS)
    assert PRICE_INDICATORS.isdisjoint(FLOW_INDICATORS)
    assert ALL_BASE_INDICATORS == FINANCIAL_INDICATORS | PRICE_INDICATORS | FLOW_INDICATORS
    assert len(FINANCIAL_INDICATORS) == 12  # README §21.1
    assert len(PRICE_INDICATORS) == 16  # README §21.2
    assert len(FLOW_INDICATORS) == 4  # README §21.3


@pytest.mark.parametrize("name", sorted(FINANCIAL_INDICATORS))
def test_financial_indicators_resolve(name: str) -> None:
    resolved = resolve_indicator(name)
    assert resolved.base == name
    assert resolved.lag == 0
    assert resolved.source is IndicatorSource.FINANCIAL


@pytest.mark.parametrize("name", sorted(PRICE_INDICATORS))
def test_price_indicators_resolve(name: str) -> None:
    resolved = resolve_indicator(name)
    assert resolved.base == name
    assert resolved.lag == 0
    assert resolved.source is IndicatorSource.PRICE


@pytest.mark.parametrize("name", sorted(FLOW_INDICATORS))
def test_flow_indicators_resolve(name: str) -> None:
    resolved = resolve_indicator(name)
    assert resolved.base == name
    assert resolved.lag == 0
    assert resolved.source is IndicatorSource.FLOW


def test_lag_suffix_on_price_indicator_readme_example() -> None:
    """README §23.4가 실제로 쓰는 lag 지표 (MILESTONES 정오표 1 해소 대상)."""
    resolved = resolve_indicator("rolling_high_60_lag1")
    assert resolved.name == "rolling_high_60_lag1"
    assert resolved.base == "rolling_high_60"
    assert resolved.lag == 1
    assert resolved.source is IndicatorSource.PRICE


def test_lag_suffix_works_generically_across_all_sources() -> None:
    """lag 문법은 PRICE 전용이 아니라 세 분류 모두에 동일하게 적용된다."""
    flow = resolve_indicator("foreign_net_buy_20d_lag2")
    assert flow.base == "foreign_net_buy_20d"
    assert flow.lag == 2
    assert flow.source is IndicatorSource.FLOW

    financial = resolve_indicator("roe_lag3")
    assert financial.base == "roe"
    assert financial.lag == 3
    assert financial.source is IndicatorSource.FINANCIAL


def test_lag_zero_is_rejected() -> None:
    """n>=1 요건 — lag0은 문법상 무의미하므로 미지원으로 거부한다."""
    with pytest.raises(StrategyValidationError):
        resolve_indicator("close_lag0")


def test_unregistered_base_is_rejected_with_hint() -> None:
    with pytest.raises(StrategyValidationError) as excinfo:
        resolve_indicator("sma_7")
    message = str(excinfo.value)
    assert "sma_7" in message
    assert "close" in message  # 허용 목록 힌트가 포함되어야 한다


def test_unregistered_lag_base_is_rejected_with_hint() -> None:
    with pytest.raises(StrategyValidationError) as excinfo:
        resolve_indicator("foo_lag1")
    message = str(excinfo.value)
    assert "foo_lag1" in message
    assert "허용" in message


def test_double_lag_suffix_is_rejected() -> None:
    """lag 중첩은 문법 밖 — base가 화이트리스트에 없으므로 명시적으로 거부된다."""
    with pytest.raises(StrategyValidationError):
        resolve_indicator("close_lag1_lag2")


def test_empty_string_is_rejected() -> None:
    with pytest.raises(StrategyValidationError):
        resolve_indicator("")
