"""강건성 분석(quant.backtest.robustness) 단위 테스트 (명세 W3c §2.1·§2.4) — 오프라인.

tmp_path에 A3/A4 정규화 스키마의 소형 parquet를 만들어 조건 분류·변형 생성(§24.3
5종 매핑, 불가 변형 skipped)·비용 배율·하위 기간 이분할 경계를 손계산 대조한다.
실데이터·네트워크·LLM은 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from research_backtest.core.hitl.models import StrategyReview
from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.robustness import RobustnessReport, run_robustness

STOCK = "000660"
CORP = "00164779"
INDEX = "1001"

BASE_CONFIG = BacktestConfig(
    commission_rate=0.00015, sell_tax_rate=0.0018, slippage_rate=0.001, initial_cash=10_000_000.0
)

# 소스별 조건: 실적(FINANCIAL)·수급(FLOW)·가격(PRICE) — 워밍업을 줄이려 짧은 지표 사용.
_FIN_COND = {"left": "operating_income_yoy", "operator": ">", "right": 0.0}
_FLOW_COND = {"left": "foreign_net_buy_5d", "operator": ">", "right": 0.0}
_PRICE_COND = {"left": "close", "operator": ">", "right": "sma_5"}
_EXIT: dict[str, Any] = {"any": [{"type": "max_holding_days", "value": 3}]}
_EXECUTION = {"signal_time": "close", "trade_time": "next_open"}


def _strategy(name: str, entry_conditions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strategy_name": name,
        "version": "1.0",
        "universe": {"type": "single_asset", "tickers": [STOCK]},
        "entry": {"all": entry_conditions},
        "exit": _EXIT,
        "execution": _EXECUTION,
    }


def _review(strategy: dict[str, Any]) -> StrategyReview:
    return StrategyReview(
        review_id="rv-robust",
        hypothesis_id="hyp-robust",
        llm_draft_strategy=strategy,
        final_strategy=strategy,
        modifications=[],
        approval_reason="강건성 단위 테스트 승인",
        approved_by="tester",
        approved_at="2026-07-15T09:00:00+09:00",
    )


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _build_data_dir(tmp_path: Path, *, n_days: int = 40) -> tuple[Path, list[date]]:
    """A3 daily·지수·A4 financial_metrics 소형 parquet 데이터셋을 만든다(거래일 리스트 반환)."""
    data_dir = tmp_path / "data"
    dates = _weekdays(date(2024, 1, 1), n_days)
    # 진입·청산이 생기도록 오르내리는 종가 + 양(+)의 외국인 순매수(FLOW 조건 True 유도)
    closes = [100 + (10 if i % 6 < 3 else -5) + i for i in range(n_days)]
    opens = [c - 1 for c in closes]

    stock_dir = data_dir / "normalized" / "market" / STOCK
    stock_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [c + 2 for c in closes],
            "low": [o - 2 for o in opens],
            "close": closes,
            "volume": [1000] * n_days,
            "foreign_net_buy_value": [1000 * (1 if i % 2 == 0 else -1) for i in range(n_days)],
            "institution_net_buy_value": [0] * n_days,
        }
    ).to_parquet(stock_dir / "daily.parquet", engine="pyarrow", index=False)

    index_dir = data_dir / "normalized" / "market" / f"index_{INDEX}"
    index_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": [2000.0] * n_days,
            "high": [2010.0] * n_days,
            "low": [1990.0] * n_days,
            "close": [2000.0 + i for i in range(n_days)],
            "volume": [1] * n_days,
            "trading_value": [1] * n_days,
        }
    ).to_parquet(index_dir / "daily.parquet", engine="pyarrow", index=False)

    fin_dir = data_dir / "normalized" / "financials" / CORP
    fin_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "metric_id": ["operating_income_yoy"],
            "fs_scope": ["CFS"],
            "fiscal_year": [2024],
            "fiscal_quarter": [1],
            "period_end": [date(2024, 1, 5)],
            "value": [0.5],  # > 0 → 실적 조건 충족
            "rcept_no": ["r1"],
            "rcept_dt": [date(2024, 1, 3)],
            "available_from": [date(2024, 1, 4)],
            "inputs_derived": [False],
        }
    ).to_parquet(fin_dir / "financial_metrics.parquet", engine="pyarrow", index=False)
    return data_dir, dates


def _run(tmp_path: Path, strategy: dict[str, Any], *, n_days: int = 40) -> RobustnessReport:
    data_dir, dates = _build_data_dir(tmp_path, n_days=n_days)
    return run_robustness(
        _review(strategy),
        data_dir=data_dir,
        stock_code=STOCK,
        corp_code=CORP,
        start_date=dates[0],
        end_date=dates[-1],
        base_config=BASE_CONFIG,
    )


# --- 조건 제거: §24.3 5종 매핑 (전 소스 존재 → 5변형 전부 구성) -------------------


def test_full_strategy_generates_all_five_ablation_variants(tmp_path: Path) -> None:
    report = _run(tmp_path, _strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND]))

    variants = [a.variant for a in report.condition_ablation]
    assert variants == [
        "가격 모멘텀만",
        "실적 모멘텀만",
        "실적 + 가격",
        "실적 + 수급",
        "실적 + 수급 + 가격",
    ]
    by_variant = {a.variant: a for a in report.condition_ablation}
    assert by_variant["가격 모멘텀만"].sources == ["PRICE"]
    assert by_variant["가격 모멘텀만"].num_conditions == 1
    assert by_variant["실적 + 수급 + 가격"].sources == ["FINANCIAL", "FLOW", "PRICE"]
    assert by_variant["실적 + 수급 + 가격"].num_conditions == 3
    # 조건 제거로 인한 skipped(불가 변형)는 없다 — §24.2 잔여 3건만 남는다.
    assert not [s for s in report.skipped if s.startswith("조건 제거")]


def test_partial_strategy_skips_unconstructible_variants(tmp_path: Path) -> None:
    """실적+가격만 있는 전략은 수급이 필요한 2변형을 skipped에 사유와 함께 남긴다(§24.3)."""
    report = _run(tmp_path, _strategy("FinPrice", [_FIN_COND, _PRICE_COND]))

    variants = [a.variant for a in report.condition_ablation]
    assert variants == ["가격 모멘텀만", "실적 모멘텀만", "실적 + 가격"]
    ablation_skips = [s for s in report.skipped if s.startswith("조건 제거")]
    assert len(ablation_skips) == 2
    assert all("수급" in s for s in ablation_skips)
    assert "실적 + 수급" in " ".join(ablation_skips)
    assert "실적 + 수급 + 가격" in " ".join(ablation_skips)


def test_price_only_strategy_yields_single_variant(tmp_path: Path) -> None:
    report = _run(tmp_path, _strategy("PriceOnly", [_PRICE_COND]))

    variants = [a.variant for a in report.condition_ablation]
    assert variants == ["가격 모멘텀만"]
    assert len([s for s in report.skipped if s.startswith("조건 제거")]) == 4


def test_full_variant_matches_base_self_check(tmp_path: Path) -> None:
    """전체 조건 변형(실적+수급+가격)이 원 전략(비용 1배)과 동일 결과여야 한다(자기 검증)."""
    report = _run(tmp_path, _strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND]))
    full = next(a for a in report.condition_ablation if a.variant == "실적 + 수급 + 가격")
    cost_1x = next(c for c in report.cost_sensitivity if c.multiplier == 1.0)
    assert full.num_trades == cost_1x.num_trades
    assert full.cumulative_return == cost_1x.cumulative_return


# --- 비용 민감도: 0배/1배/2배 (§24.2) ---------------------------------------


def test_cost_sensitivity_multipliers_and_scaling(tmp_path: Path) -> None:
    report = _run(tmp_path, _strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND]))

    assert [c.multiplier for c in report.cost_sensitivity] == [0.0, 1.0, 2.0]
    zero, one, two = report.cost_sensitivity
    # 0배 = 무비용
    assert zero.commission_rate == 0.0 and zero.sell_tax_rate == 0.0 and zero.slippage_rate == 0.0
    # 1배 = base_config 그대로
    assert one.commission_rate == pytest.approx(BASE_CONFIG.commission_rate)
    assert one.sell_tax_rate == pytest.approx(BASE_CONFIG.sell_tax_rate)
    assert one.slippage_rate == pytest.approx(BASE_CONFIG.slippage_rate)
    # 2배
    assert two.commission_rate == pytest.approx(BASE_CONFIG.commission_rate * 2)
    assert two.sell_tax_rate == pytest.approx(BASE_CONFIG.sell_tax_rate * 2)
    assert two.slippage_rate == pytest.approx(BASE_CONFIG.slippage_rate * 2)
    # 거래 수는 비용과 무관하게 동일(신호가 같으므로) — 수익률만 비용에 따라 달라진다.
    assert zero.num_trades == one.num_trades == two.num_trades


# --- 하위 기간: 이분할 경계 (§24.2) -----------------------------------------


def test_subperiod_bisection_boundary(tmp_path: Path) -> None:
    """[start, mid]·[mid, end] 이분할, mid = 거래일 기준 중앙(index[n//2])."""
    data_dir, dates = _build_data_dir(tmp_path, n_days=40)
    report = run_robustness(
        _review(_strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND])),
        data_dir=data_dir,
        stock_code=STOCK,
        corp_code=CORP,
        start_date=dates[0],
        end_date=dates[-1],
        base_config=BASE_CONFIG,
    )
    assert len(report.subperiod) == 2
    first, second = report.subperiod
    mid = dates[len(dates) // 2]
    assert first.label == "전반부"
    assert first.start_date == dates[0]
    assert first.end_date == mid
    assert second.label == "후반부"
    assert second.start_date == mid
    assert second.end_date == dates[-1]


def test_subperiod_skipped_when_too_few_trading_days(tmp_path: Path) -> None:
    """거래일이 2일 미만이면 하위 기간 분석을 skipped에 사유와 함께 남긴다."""
    data_dir, dates = _build_data_dir(tmp_path, n_days=40)
    single_day = dates[10]
    report = run_robustness(
        _review(_strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND])),
        data_dir=data_dir,
        stock_code=STOCK,
        corp_code=CORP,
        start_date=single_day,
        end_date=single_day,
        base_config=BASE_CONFIG,
    )
    assert report.subperiod == []
    assert any("하위 기간" in s for s in report.skipped)


# --- §24.2 잔여 항목: 조용한 누락 금지 --------------------------------------


def test_deferred_robustness_items_recorded_in_skipped(tmp_path: Path) -> None:
    report = _run(tmp_path, _strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND]))
    joined = " ".join(report.skipped)
    assert "인샘플/아웃오브샘플" in joined
    assert "파라미터 민감도" in joined
    assert "시장 국면" in joined


def test_missing_data_raises_file_not_found(tmp_path: Path) -> None:
    """daily.parquet 부재 시 FileNotFoundError(호출부가 exit 1로 매핑)."""
    with pytest.raises(FileNotFoundError):
        run_robustness(
            _review(_strategy("Full", [_FIN_COND, _FLOW_COND, _PRICE_COND])),
            data_dir=tmp_path / "empty",
            stock_code=STOCK,
            corp_code=CORP,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 15),
            base_config=BASE_CONFIG,
        )
