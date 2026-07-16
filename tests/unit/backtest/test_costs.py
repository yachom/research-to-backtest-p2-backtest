"""costs.py — configs/backtest.yaml 로더 테스트 (명세 A6 §5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_backtest.core.exceptions import ConfigError
from research_backtest.quant.backtest.costs import (
    DEFAULT_BACKTEST_CONFIG_PATH,
    BacktestConfig,
    load_backtest_config,
)

REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "backtest.yaml"


def test_load_repo_config_values() -> None:
    """레포 configs/backtest.yaml 값을 그대로 읽는다 (README §23~§24)."""
    config = load_backtest_config(REPO_CONFIG)
    assert config.commission_rate == pytest.approx(0.00015)
    assert config.sell_tax_rate == pytest.approx(0.0018)
    assert config.slippage_rate == pytest.approx(0.001)
    assert config.initial_cash == pytest.approx(100_000_000)
    assert config.benchmark == "KOSPI"
    assert config.strategy_style == "long_cash"
    assert config.signal_time == "close"
    assert config.trade_time == "next_open"


def test_default_path_points_to_repo_config() -> None:
    """기본 경로 상수가 configs/backtest.yaml을 가리킨다."""
    assert Path("configs/backtest.yaml") == DEFAULT_BACKTEST_CONFIG_PATH


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="백테스트 설정 파일이 없습니다"):
        load_backtest_config(tmp_path / "nope.yaml")


def test_missing_sections_use_defaults(tmp_path: Path) -> None:
    """섹션이 비어도 README 기본값으로 채운다."""
    path = tmp_path / "backtest.yaml"
    path.write_text("costs:\n  commission_rate: 0.0005\n", encoding="utf-8")
    config = load_backtest_config(path)
    assert config.commission_rate == pytest.approx(0.0005)
    assert config.sell_tax_rate == pytest.approx(0.0018)  # 기본값
    assert config.initial_cash == pytest.approx(100_000_000)  # 기본값


def test_negative_rate_rejected(tmp_path: Path) -> None:
    path = tmp_path / "backtest.yaml"
    path.write_text("costs:\n  commission_rate: -0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="백테스트 설정 값이 잘못"):
        load_backtest_config(path)


def test_non_mapping_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "backtest.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="매핑이 아님"):
        load_backtest_config(path)


def test_invalid_trade_time_rejected(tmp_path: Path) -> None:
    """룩어헤드 방지 — trade_time은 next_open 고정, 다른 값 거부."""
    path = tmp_path / "backtest.yaml"
    path.write_text("execution:\n  trade_time: same_close\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_backtest_config(path)


def test_config_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match=r"[Ee]xtra"):
        BacktestConfig.model_validate(
            {
                "commission_rate": 0.0,
                "sell_tax_rate": 0.0,
                "slippage_rate": 0.0,
                "initial_cash": 1.0,
                "unknown_field": 1,
            }
        )
