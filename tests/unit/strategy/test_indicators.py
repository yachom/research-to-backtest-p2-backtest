"""지표 계산 테스트 — 손계산 대조·Wilder 공식 스모크·no-lookahead (명세 A5 §4·§6).

no-lookahead 항목이 DoD 3의 "지표 레벨" 룩어헤드 테스트다(README §28.3 취지):
프레임을 t까지 잘라도 0..t행의 지표 값은 전체 프레임으로 계산한 값과
동일해야 한다 — rolling/ewm 등 인과적 연산만 썼다면 항상 성립해야 하는
불변량이므로 property 테스트로 여러 지표·여러 절단점에 대해 검증한다.
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.quant.strategy.indicators import compute_indicators
from research_backtest.quant.strategy.registry import (
    ALL_BASE_INDICATORS,
    IndicatorSource,
    resolve_indicator,
)

# conftest.py의 make_daily_frame 픽스처 반환 타입 — 디렉토리에 __init__.py가 없어
# cross-module import 대신 각 테스트 파일에서 동일하게 선언한다(명세 A5 §6).
DailyFrameFactory = Callable[..., pd.DataFrame]

# --- 손계산 대조 -------------------------------------------------------------


def test_sma_hand_calculation(make_daily_frame: DailyFrameFactory) -> None:
    close = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"sma_5"})

    expected = [math.nan, math.nan, math.nan, math.nan, 12.0, 13.0, 14.0]
    got = out["sma_5"].tolist()
    for e, g in zip(expected, got, strict=True):
        if math.isnan(e):
            assert math.isnan(g)
        else:
            assert g == pytest.approx(e)


def test_rolling_high_includes_today_lag1_excludes_today(
    make_daily_frame: DailyFrameFactory,
) -> None:
    """rolling_high_N(당일 포함) vs _lag1(당일 제외, 직전 N일) 구분 — README §23.4 각주.

    rolling_high_20은 min_periods=20이므로 스파이크를 윈도우가 이미 찬(19번째
    행 이후) 시점에 넣어야 당일 포함/lag1의 차이를 관찰할 수 있다.
    """
    n = 25
    high = [10.0] * n
    high[20] = 30.0  # 윈도우가 이미 찬 뒤(day19까지 워밍업 완료)에 스파이크
    daily = make_daily_frame(close=high, high=high)

    out = compute_indicators(daily, {"rolling_high_20", "rolling_high_20_lag1"})

    assert out["rolling_high_20"].iloc[19] == pytest.approx(10.0)  # 첫 유효값(스파이크 이전)
    # 당일 포함 버전: day20부터 스파이크(30.0)를 즉시 반영한다.
    assert out["rolling_high_20"].iloc[20] == pytest.approx(30.0)
    # lag1(직전 값): day20 시점에는 아직 어제(day19)까지의 최고치(10.0) — 스파이크 미반영.
    assert out["rolling_high_20_lag1"].iloc[20] == pytest.approx(10.0)
    # day21에서야 lag1이 스파이크를 반영한다(하루 지연).
    assert out["rolling_high_20_lag1"].iloc[21] == pytest.approx(30.0)
    # lag1은 정확히 당일 포함 버전의 shift(1)과 같다.
    pd.testing.assert_series_equal(
        out["rolling_high_20_lag1"], out["rolling_high_20"].shift(1), check_names=False
    )


def test_return_nd_hand_calculation(make_daily_frame: DailyFrameFactory) -> None:
    close = [100.0, 110.0, 121.0, 108.9, 130.0]
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"return_20d"})
    # N=20 > 데이터 길이이므로 전부 NaN(워밍업 미충족) — 채우지 않는다.
    assert out["return_20d"].isna().all()


def test_return_nd_hand_calculation_with_valid_window(make_daily_frame: DailyFrameFactory) -> None:
    close = [100.0 + i for i in range(25)]  # N=20 워밍업을 넘기는 길이
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"return_20d"})

    assert out["return_20d"].iloc[:20].isna().all()
    expected_20 = close[20] / close[0] - 1.0
    expected_24 = close[24] / close[4] - 1.0
    assert out["return_20d"].iloc[20] == pytest.approx(expected_20)
    assert out["return_20d"].iloc[24] == pytest.approx(expected_24)


def test_foreign_institution_rolling_sum(make_daily_frame: DailyFrameFactory) -> None:
    close = [1.0] * 6
    foreign = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    institution = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    daily = make_daily_frame(
        close=close, foreign_net_buy_value=foreign, institution_net_buy_value=institution
    )

    out = compute_indicators(daily, {"foreign_net_buy_5d", "institution_net_buy_5d"})

    assert out["foreign_net_buy_5d"].iloc[:4].isna().all()
    assert out["foreign_net_buy_5d"].iloc[4] == pytest.approx(sum(foreign[0:5]))
    assert out["foreign_net_buy_5d"].iloc[5] == pytest.approx(sum(foreign[1:6]))
    assert out["institution_net_buy_5d"].iloc[4] == pytest.approx(sum(institution[0:5]))


# --- Wilder RSI/ATR 스모크 ----------------------------------------------------


def test_rsi_wilder_all_gains_approaches_100(make_daily_frame: DailyFrameFactory) -> None:
    close = [100.0 + i for i in range(30)]  # 매일 상승 — 손실 없음
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"rsi_14"})

    assert out["rsi_14"].iloc[:14].isna().all()  # 워밍업 14개는 NaN
    assert out["rsi_14"].iloc[14:].notna().all()
    assert (out["rsi_14"].iloc[14:] > 99.0).all()  # avg_loss=0 -> RSI -> 100


def test_rsi_wilder_all_losses_approaches_0(make_daily_frame: DailyFrameFactory) -> None:
    close = [200.0 - i for i in range(30)]  # 매일 하락 — 이익 없음
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"rsi_14"})

    assert (out["rsi_14"].iloc[14:] < 1.0).all()  # avg_gain=0 -> RSI -> 0


def test_rsi_wilder_flat_series_is_nan(make_daily_frame: DailyFrameFactory) -> None:
    """등락이 전혀 없으면 avg_gain=avg_loss=0 -> RS=0/0 -> RSI는 NaN(정의역 밖)."""
    close = [100.0] * 20
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"rsi_14"})
    assert out["rsi_14"].iloc[14:].isna().all()


def test_atr_wilder_hand_calculation_first_valid_value(make_daily_frame: DailyFrameFactory) -> None:
    """TR 정의(첫 행은 high-low만) + Wilder 재귀를 직접 검산한다 — README §21.2 atr_14.

    atr_14는 min_periods=14라 13번째 행까지는 NaN이다. TR을 처음 13개 행에서
    상수(2.0)로 고정하면 ewm(adjust=False) 재귀가 그 상수에 머무르므로
    (EMA of a constant = the constant), 14번째 행(index13)의 값만 손으로
    풀어도 정확히 검증할 수 있다: atr[13] = (13/14)*2.0 + (1/14)*TR[13].
    """
    n = 14
    high = [101.0] * (n - 1) + [110.0]
    low = [99.0] * (n - 1) + [100.0]
    close = [100.0] * (n - 1) + [105.0]
    daily = make_daily_frame(close=close, high=high, low=low)

    out = compute_indicators(daily, {"atr_14"})

    assert out["atr_14"].iloc[:13].isna().all()
    # TR[0..12] = high-low = 2.0(상수). TR[13] = max(110-100, |110-100|, |100-100|) = 10.0.
    expected_13 = (13 / 14) * 2.0 + (1 / 14) * 10.0
    assert out["atr_14"].iloc[13] == pytest.approx(expected_13)


def test_atr_warmup_matches_min_periods(make_daily_frame: DailyFrameFactory) -> None:
    n = 20
    rng = np.random.default_rng(1)
    close = list(100 + np.cumsum(rng.normal(0, 1, n)))
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    daily = make_daily_frame(close=close, high=high, low=low)

    out = compute_indicators(daily, {"atr_14"})
    assert out["atr_14"].iloc[:13].isna().all()
    assert out["atr_14"].iloc[13:].notna().all()


# --- 워밍업 NaN 유지 (채우지 않음) -------------------------------------------


def test_warmup_region_is_nan_not_filled(make_daily_frame: DailyFrameFactory) -> None:
    close = list(range(1, 25))
    daily = make_daily_frame(close=[float(c) for c in close])

    out = compute_indicators(daily, {"sma_20", "rolling_high_20"})

    assert out["sma_20"].iloc[:19].isna().all()
    assert out["rolling_high_20"].iloc[:19].isna().all()
    assert out["sma_20"].iloc[19:].notna().all()


# --- lag는 임의 지표에 범용 적용 ---------------------------------------------


def test_lag_is_generic_shift_of_base(make_daily_frame: DailyFrameFactory) -> None:
    close = [float(10 + i) for i in range(10)]
    daily = make_daily_frame(close=close)

    out = compute_indicators(daily, {"sma_5", "sma_5_lag3"})
    pd.testing.assert_series_equal(out["sma_5_lag3"], out["sma_5"].shift(3), check_names=False)


# --- 소스 경계: FINANCIAL은 여기서 계산하지 않는다 ---------------------------


def test_financial_indicator_is_not_computed(make_daily_frame: DailyFrameFactory) -> None:
    daily = make_daily_frame(close=[100.0, 101.0, 102.0])
    out = compute_indicators(daily, {"roe"})
    assert "roe" not in out.columns  # A6가 as-of join으로 공급 — 여기서 계산/요구하지 않는다


def test_lagged_financial_requires_preexisting_base_column(
    make_daily_frame: DailyFrameFactory,
) -> None:
    daily = make_daily_frame(close=[100.0, 101.0, 102.0, 103.0])

    with pytest.raises(StrategyValidationError):
        compute_indicators(daily, {"roe_lag1"})

    pre_joined = daily.copy()
    pre_joined["roe"] = [0.1, 0.1, 0.2, 0.2]
    out = compute_indicators(pre_joined, {"roe_lag1"})
    pd.testing.assert_series_equal(out["roe_lag1"], pre_joined["roe"].shift(1), check_names=False)


# --- 오류 처리 ---------------------------------------------------------------


def test_unsupported_indicator_raises(make_daily_frame: DailyFrameFactory) -> None:
    daily = make_daily_frame(close=[1.0, 2.0, 3.0])
    with pytest.raises(StrategyValidationError):
        compute_indicators(daily, {"sma_7"})


def test_missing_daily_schema_column_raises() -> None:
    daily = pd.DataFrame({"close": [1.0, 2.0, 3.0]})  # high 없음
    with pytest.raises(StrategyValidationError):
        compute_indicators(daily, {"rolling_high_20"})


def test_does_not_mutate_input_frame(make_daily_frame: DailyFrameFactory) -> None:
    daily = make_daily_frame(close=[float(c) for c in range(1, 12)])
    original_columns = set(daily.columns)

    compute_indicators(daily, {"sma_5"})

    assert set(daily.columns) == original_columns  # 입력 프레임은 그대로다(사본 반환)


# --- no-lookahead property 테스트 (DoD 3, README §28.3 취지의 지표 레벨) ----


def _synthetic_daily(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1.2, n))
    close = np.maximum(close, 5.0)
    high = close + rng.uniform(0.0, 1.5, n)
    low = close - rng.uniform(0.0, 1.5, n)
    open_ = close + rng.normal(0.0, 0.3, n)
    volume = rng.integers(1_000, 5_000, n).astype(float)
    foreign = rng.normal(0, 1_000, n)
    institution = rng.normal(0, 800, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "foreign_net_buy_value": foreign,
            "institution_net_buy_value": institution,
        }
    ).set_index("date")


_PROPERTY_TEST_INDICATORS = sorted(
    {
        "sma_5",
        "sma_20",
        "rolling_high_60",
        "rolling_high_60_lag1",
        "return_20d",
        "volatility_20",
        "rsi_14",
        "atr_14",
        "foreign_net_buy_20d",
        "institution_net_buy_5d",
    }
)


@pytest.mark.parametrize("cutoff", [15, 30, 61, 90, 119])
def test_no_lookahead_truncation_invariance(cutoff: int) -> None:
    """프레임을 cutoff까지 잘라도 0..cutoff-1행의 지표 값은 전체 프레임과 동일해야 한다.

    rolling/ewm/pct_change/shift는 전부 인과적(과거→현재) 연산이므로, cutoff
    이후 행이 존재하든 말든 cutoff 이전 행의 값에는 영향이 없어야 한다 —
    이것이 깨지면 미래 정보가 과거 신호 계산에 스며든 것(룩어헤드)이다.
    """
    full_daily = _synthetic_daily(n=150, seed=20260714)
    required = set(_PROPERTY_TEST_INDICATORS)

    full = compute_indicators(full_daily, required)
    truncated = compute_indicators(full_daily.iloc[:cutoff], required)

    for name in required:
        full_prefix = full[name].iloc[:cutoff].to_numpy(dtype=float)
        truncated_values = truncated[name].to_numpy(dtype=float)
        assert np.array_equal(full_prefix, truncated_values, equal_nan=True), (
            f"{name}: 프레임 절단으로 0..{cutoff - 1}행 값이 바뀌었습니다(룩어헤드 의심)."
        )


def test_no_lookahead_future_mutation_invariance() -> None:
    """t+1행 이후 값을 바꿔도 0..t행의 지표 값은 변하지 않아야 한다(절단과 상보적인 확인)."""
    base = _synthetic_daily(n=100, seed=7)
    t = 50
    required = set(_PROPERTY_TEST_INDICATORS)

    before = compute_indicators(base, required)

    mutated = base.copy()
    rng = np.random.default_rng(999)
    future_slice = mutated.iloc[t + 1 :]
    future_n = len(future_slice)
    mutated.loc[future_slice.index, "close"] = rng.normal(1_000, 50, future_n)
    mutated.loc[future_slice.index, "high"] = mutated.loc[future_slice.index, "close"] + 5
    mutated.loc[future_slice.index, "low"] = mutated.loc[future_slice.index, "close"] - 5
    mutated.loc[future_slice.index, "foreign_net_buy_value"] = rng.normal(0, 5_000, future_n)
    mutated.loc[future_slice.index, "institution_net_buy_value"] = rng.normal(0, 5_000, future_n)

    after = compute_indicators(mutated, required)

    for name in required:
        a = before[name].iloc[: t + 1].to_numpy(dtype=float)
        b = after[name].iloc[: t + 1].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), f"{name}: 미래 값 변경이 과거 값을 바꿨습니다."


def test_all_price_and_flow_indicators_are_computable() -> None:
    """레지스트리의 PRICE·FLOW 화이트리스트 전부가 실제로 계산 정의를 갖는지 확인한다
    (레지스트리·계산기 drift 방지)."""
    price_and_flow = {
        name
        for name in ALL_BASE_INDICATORS
        if resolve_indicator(name).source is not IndicatorSource.FINANCIAL
    }
    daily = _synthetic_daily(n=130, seed=3)
    out = compute_indicators(daily, price_and_flow)
    for name in price_and_flow:
        assert name in out.columns
