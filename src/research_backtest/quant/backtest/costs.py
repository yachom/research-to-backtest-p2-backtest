"""백테스트 비용·실행 설정 — configs/backtest.yaml 로더 (명세 A6 §5, README §23~§24).

``configs/backtest.yaml``는 A6가 **소비만** 하는 공유 설정이다(명세 A6 §0). 이
모듈은 그 YAML을 :class:`BacktestConfig`(수수료·세금·슬리피지·초기자본·벤치마크·
실행 규칙)로 검증해 엔진·러너에 넘긴다. 파싱 실패·값 오류는
:class:`ConfigError`로 통일한다(core.config.load_market_config 패턴).

체결 규칙(README §23.3)은 MVP에서 값이 고정이다 — 신호는 t일 종가로 계산하고
체결은 t+1 거래일 시가에 한다. 스키마가 다른 값을 거부해 룩어헤드 규칙이
설정으로 뒤집히지 않게 한다(A5 ExecutionSpec과 동일 취지).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_backtest.core.exceptions import ConfigError

DEFAULT_BACKTEST_CONFIG_PATH = Path("configs/backtest.yaml")


class BacktestConfig(BaseModel):
    """백테스트 비용·실행 파라미터 (configs/backtest.yaml, README §23.3·§24).

    - ``commission_rate``: 위탁수수료(편도, 매수·매도 각각 체결금액에 적용).
    - ``sell_tax_rate``: 증권거래세+농특세 — **매도 금액에만** 적용(명세 A6 §3).
    - ``slippage_rate``: 매수는 ``open*(1+slippage)``, 매도는 ``open*(1-slippage)``.
    - ``initial_cash``: 초기 자본(원).
    - ``signal_time``/``trade_time``: 체결 타이밍 — 값 고정(룩어헤드 방지).
    """

    model_config = ConfigDict(extra="forbid")

    commission_rate: float = Field(ge=0.0)
    sell_tax_rate: float = Field(ge=0.0)
    slippage_rate: float = Field(ge=0.0)
    initial_cash: float = Field(gt=0.0)

    benchmark: str = "KOSPI"
    strategy_style: str = "long_cash"

    signal_time: Literal["close"] = "close"
    trade_time: Literal["next_open"] = "next_open"


def load_backtest_config(path: Path = DEFAULT_BACKTEST_CONFIG_PATH) -> BacktestConfig:
    """configs/backtest.yaml을 읽어 :class:`BacktestConfig`로 검증한다.

    YAML은 ``execution``·``costs``·``defaults`` 세 섹션으로 나뉘어 있으나
    (configs/backtest.yaml), :class:`BacktestConfig`는 이를 평평한 필드로
    합쳐 보관한다. 파일 부재·형식 오류·값 오류는 :class:`ConfigError`로
    통일한다(core.config 로더 패턴).
    """
    if not path.exists():
        raise ConfigError(f"백테스트 설정 파일이 없습니다: {path} (레포 루트에서 실행했는지 확인)")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"백테스트 설정 파일 형식이 잘못되었습니다(매핑이 아님): {path}")

    costs = _section(raw, "costs", path)
    execution = _section(raw, "execution", path)
    defaults = _section(raw, "defaults", path)

    try:
        return BacktestConfig.model_validate(
            {
                "commission_rate": costs.get("commission_rate", 0.00015),
                "sell_tax_rate": costs.get("sell_tax_rate", 0.0018),
                "slippage_rate": costs.get("slippage_rate", 0.001),
                "initial_cash": defaults.get("initial_cash", 100_000_000),
                "benchmark": defaults.get("benchmark", "KOSPI"),
                "strategy_style": defaults.get("strategy_style", "long_cash"),
                "signal_time": execution.get("signal_time", "close"),
                "trade_time": execution.get("trade_time", "next_open"),
            }
        )
    except ValidationError as err:
        raise ConfigError(f"백테스트 설정 값이 잘못되었습니다: {err}") from err


def _section(raw: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    """최상위 섹션(costs/execution/defaults)을 꺼낸다 — 없으면 빈 매핑, 매핑 아님이면 오류."""
    section: Any = raw.get(name) or {}
    if not isinstance(section, dict):
        raise ConfigError(f"백테스트 설정의 {name} 항목이 매핑이 아닙니다: {path}")
    return section
