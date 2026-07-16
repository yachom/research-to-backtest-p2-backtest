"""전략 DSL(quant.strategy) 단위 테스트 공용 픽스처 (명세 A5 §6) — 전부 오프라인."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "strategy"
EARNINGS_FLOW_BREAKOUT_PATH = FIXTURE_DIR / "earnings_flow_breakout.json"

DailyFrameFactory = Callable[..., pd.DataFrame]


@pytest.fixture
def earnings_flow_breakout_path() -> Path:
    """README §23.4 전략 JSON 원본 fixture 경로 (수정 없이 그대로 검증·컴파일)."""
    return EARNINGS_FLOW_BREAKOUT_PATH


@pytest.fixture
def earnings_flow_breakout_raw() -> dict[str, Any]:
    """README §23.4 전략 JSON을 dict로 로드한다.

    호출마다 파일을 새로 읽어 반환하므로, 테스트가 dict를 변형(잘못된 케이스
    생성 등)해도 다른 테스트에 영향을 주지 않는다.
    """
    return dict(json.loads(EARNINGS_FLOW_BREAKOUT_PATH.read_text(encoding="utf-8")))


@pytest.fixture
def make_daily_frame() -> DailyFrameFactory:
    """A3 daily.parquet 스키마의 소형 DataFrame 팩토리 (명세 A5 §4 손계산 테스트용).

    ``close``만 넘기면 open/high/low가 close와 같고 volume·수급이 0인 안전한
    기본 프레임을 만든다 — rolling_high 등 high/low가 의미 있는 케이스는
    해당 인자를 명시한다. index는 ``date``(오름차순 영업일)다.
    """

    def _make(
        *,
        close: list[float],
        high: list[float] | None = None,
        low: list[float] | None = None,
        open_: list[float] | None = None,
        volume: list[float] | None = None,
        foreign_net_buy_value: list[float] | None = None,
        institution_net_buy_value: list[float] | None = None,
        start: str = "2024-01-02",
    ) -> pd.DataFrame:
        n = len(close)
        dates = pd.bdate_range(start, periods=n)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": open_ if open_ is not None else close,
                "high": high if high is not None else close,
                "low": low if low is not None else close,
                "close": close,
                "volume": volume if volume is not None else [1_000.0] * n,
                "foreign_net_buy_value": (
                    foreign_net_buy_value if foreign_net_buy_value is not None else [0.0] * n
                ),
                "institution_net_buy_value": (
                    institution_net_buy_value
                    if institution_net_buy_value is not None
                    else [0.0] * n
                ),
            }
        ).set_index("date")
        return frame

    return _make
