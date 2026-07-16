"""강건성 분석 — 조건 제거·비용 민감도·하위 기간 (README §24.2·§24.3, 명세 W3c §2.1).

승인된 전략(:class:`StrategyReview.final_strategy`)을 여러 방식으로 변형·재실행해
성과의 강건성을 점검한다. 변형은 **승인본이 아니므로** ``execute_approved_strategy``
(승인 게이트·산출물 저장)를 거치지 않고 :func:`engine.run_backtest` 연구용 경로를
쓴다 — runner docstring이 허용하는 "테스트·연구용" 경로다(명세 W3c §2.1). review는
변조하지 않는다.

세 가지 분석(§24.2에서 이번 범위):

- **조건 제거(§24.3)**: ``entry.all`` 조건을 소스(실적/가격/수급)로 분류하고, §24.3의
  5개 변형 중 원 전략의 조건 구성으로 만들 수 있는 것만 재실행한다. 불가능한 변형·
  컴파일 실패는 ``skipped``에 사유를 남긴다(조용한 누락 금지).
- **비용 민감도**: commission·sell_tax·slippage를 0배/1배/2배 배율로 원 전략 재실행.
  1배 결과는 승인 백테스트와 동일해야 한다(``base_config``·기간이 같으므로) — 이를
  자기 검증에 쓴다.
- **하위 기간**: [start, mid]·[mid, end] 이분할(mid=거래일 기준 중앙) 원 전략 재실행.

§24.2의 나머지(인샘플/아웃오브샘플·파라미터 민감도·시장 국면)는 ``skipped``에
"후순위(제출 후 확장)"로 기록한다.

데이터 로드는 runner의 관례(A3 daily·A4 financial_metrics·지수 parquet)를 따르되,
전 기간 as-of join(``build_backtest_frame``)은 전략과 무관하므로 한 번만 수행해
변형·비용 재실행에서 재사용한다(중복 로드·조인 회피).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from research_backtest.core.exceptions import StrategyValidationError
from research_backtest.core.hitl.models import StrategyReview
from research_backtest.core.market.calendar import as_date
from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.data import (
    assert_no_lookahead,
    build_backtest_frame,
    truncate_to_window,
)
from research_backtest.quant.backtest.engine import run_backtest
from research_backtest.quant.backtest.metrics import BacktestResult, compute_backtest_metrics
from research_backtest.quant.strategy.compiler import (
    PositionRules,
    compile_strategy,
    entry_signal,
    exit_signal,
)
from research_backtest.quant.strategy.indicators import compute_indicators
from research_backtest.quant.strategy.registry import IndicatorSource, resolve_indicator
from research_backtest.quant.strategy.schema import (
    Condition,
    ConditionGroup,
    StrategySpec,
    parse_strategy_spec,
)

logger = logging.getLogger("r2b.backtest.robustness")

# 벤치마크명 → 지수 종목코드 (runner.BENCHMARK_INDEX_CODES와 동일 규약 — README §24).
_BENCHMARK_INDEX_CODES = {"KOSPI": "1001", "KOSDAQ": "2001"}

# 비용 민감도 배율(§24.2) — 0배(무비용)·1배(승인본과 동일)·2배(2배 비용).
_COST_MULTIPLIERS: tuple[float, ...] = (0.0, 1.0, 2.0)

# IndicatorSource → 한국어 라벨(§24.3 조건 제거 서술·skipped 사유).
_SOURCE_KO: dict[IndicatorSource, str] = {
    IndicatorSource.FINANCIAL: "실적",
    IndicatorSource.PRICE: "가격",
    IndicatorSource.FLOW: "수급",
}

# §24.3 5개 변형: (한국어 라벨, 허용 소스 집합). 변형 전략은 entry.all에서 소스가
# 이 집합의 부분집합인 조건만 남긴다. 원 전략에 각 소스 조건이 실제로 있어야
# 구성 가능하다(covered == allowed).
_ABLATION_VARIANTS: list[tuple[str, frozenset[IndicatorSource]]] = [
    ("가격 모멘텀만", frozenset({IndicatorSource.PRICE})),
    ("실적 모멘텀만", frozenset({IndicatorSource.FINANCIAL})),
    ("실적 + 가격", frozenset({IndicatorSource.FINANCIAL, IndicatorSource.PRICE})),
    ("실적 + 수급", frozenset({IndicatorSource.FINANCIAL, IndicatorSource.FLOW})),
    (
        "실적 + 수급 + 가격",
        frozenset({IndicatorSource.FINANCIAL, IndicatorSource.FLOW, IndicatorSource.PRICE}),
    ),
]

# §24.2 잔여 항목 — 이번 범위 밖(조용한 누락 금지, skipped에 명시).
_DEFERRED_ROBUSTNESS: tuple[str, ...] = (
    "인샘플/아웃오브샘플 분리 — 후순위(제출 후 확장)",
    "파라미터 민감도 — 후순위(제출 후 확장)",
    "시장 국면 분석 — 후순위(제출 후 확장)",
)


class AblationResult(BaseModel):
    """조건 제거 변형 1건의 성과 요약 (§24.3)."""

    model_config = ConfigDict(extra="forbid")

    variant: str  # "가격 모멘텀만" 등 §24.3 라벨
    sources: list[str]  # 변형에 남긴 조건 소스(FINANCIAL/PRICE/FLOW)
    num_conditions: int  # 변형 entry.all 조건 수
    num_trades: int
    cumulative_return: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None


class CostSensitivityResult(BaseModel):
    """비용 배율 1건의 성과 요약 (§24.2 거래비용 민감도)."""

    model_config = ConfigDict(extra="forbid")

    multiplier: float  # 0.0 / 1.0 / 2.0
    commission_rate: float
    sell_tax_rate: float
    slippage_rate: float
    num_trades: int
    cumulative_return: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None


class SubperiodResult(BaseModel):
    """하위 기간 1건의 성과 요약 (§24.2 하위 기간 분석)."""

    model_config = ConfigDict(extra="forbid")

    label: str  # "전반부" / "후반부"
    start_date: date
    end_date: date
    num_trades: int
    cumulative_return: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None


class RobustnessReport(BaseModel):
    """강건성 분석 산출물 — robustness_report.json 본문 (명세 W3c §2.1)."""

    model_config = ConfigDict(extra="forbid")

    strategy_name: str
    start_date: date
    end_date: date

    condition_ablation: list[AblationResult]  # §24.3 조건 제거
    cost_sensitivity: list[CostSensitivityResult]
    subperiod: list[SubperiodResult]
    skipped: list[str]  # 미수행 항목·사유 (§24.2 잔여)


# --- 신호 준비 결과(전략별로 재계산; 비용 재실행은 재사용) -----------------------


class _PreparedSignals:
    """한 전략을 한 기간에 대해 준비한 프레임·신호·청산 규칙 — 비용 배율 재실행에 재사용."""

    __slots__ = ("entry", "exit_", "frame", "position_rules", "strategy_name")

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        entry: pd.Series,
        exit_: pd.Series,
        position_rules: PositionRules,
        strategy_name: str,
    ) -> None:
        self.frame = frame
        self.entry = entry
        self.exit_ = exit_
        self.position_rules = position_rules
        self.strategy_name = strategy_name


def run_robustness(
    review: StrategyReview,
    *,
    data_dir: Path,
    stock_code: str,
    corp_code: str,
    start_date: date,
    end_date: date,
    base_config: BacktestConfig,
    fs_scope: str = "CFS",
) -> RobustnessReport:
    """승인 전략을 변형·재실행해 강건성 리포트를 만든다 (명세 W3c §2.1, README §24.2·§24.3).

    ``base_config``·[start, end]는 승인 백테스트와 동일해야 한다(호출부가 보장) —
    이 경우 비용 1배 결과·전체 조건 변형이 승인 백테스트와 일치한다. 전체 조건 변형이
    구성 가능하면 그 결과가 비용 1배 결과와 일치하는지 내부에서 assert한다(연구용 경로가
    원 전략을 그대로 재현하는지에 대한 자기 검증).
    """
    daily = _load_daily(data_dir, stock_code)
    metrics = _load_metrics(data_dir, corp_code)
    benchmark_series = _load_benchmark_series(data_dir, base_config.benchmark)

    base_spec = parse_strategy_spec(review.final_strategy)

    # 전 기간 as-of join은 전략과 무관 — 한 번만 수행해 변형·비용 재실행이 재사용한다.
    joined_full = build_backtest_frame(
        daily, metrics, fs_scope=fs_scope, start_date=start_date, end_date=end_date
    )
    assert_no_lookahead(joined_full)

    base_prepared = _prepare_signals(base_spec, joined_full, start_date, end_date)
    base_result = _run_engine(
        base_prepared, base_config, benchmark_series, start_date, end_date, fs_scope
    )

    skipped: list[str] = []
    cost_sensitivity = _run_cost_sensitivity(
        base_prepared, base_config, benchmark_series, start_date, end_date, fs_scope
    )
    condition_ablation = _run_condition_ablation(
        base_spec,
        joined_full,
        base_config,
        benchmark_series,
        start_date,
        end_date,
        fs_scope,
        base_result,
        skipped,
    )
    subperiod = _run_subperiods(
        base_spec,
        base_prepared.frame,
        daily,
        metrics,
        base_config,
        benchmark_series,
        start_date,
        end_date,
        fs_scope,
        skipped,
    )

    skipped.extend(_DEFERRED_ROBUSTNESS)

    return RobustnessReport(
        strategy_name=base_spec.strategy_name,
        start_date=start_date,
        end_date=end_date,
        condition_ablation=condition_ablation,
        cost_sensitivity=cost_sensitivity,
        subperiod=subperiod,
        skipped=skipped,
    )


# --- 조건 제거 (§24.3) -------------------------------------------------------


def _run_condition_ablation(
    base_spec: StrategySpec,
    joined_full: pd.DataFrame,
    base_config: BacktestConfig,
    benchmark_series: pd.Series | None,
    start_date: date,
    end_date: date,
    fs_scope: str,
    base_result: BacktestResult,
    skipped: list[str],
) -> list[AblationResult]:
    """§24.3의 5개 변형 중 구성 가능한 것만 재실행한다(불가·컴파일 실패는 skipped)."""
    entry_items = _entry_items(base_spec)
    item_sources = [(item, _condition_sources(item)) for item in entry_items]
    present_sources: frozenset[IndicatorSource] = frozenset(
        source for _, srcs in item_sources for source in srcs
    )

    results: list[AblationResult] = []
    for label, allowed in _ABLATION_VARIANTS:
        selected = [item for item, srcs in item_sources if srcs and srcs <= allowed]
        covered: frozenset[IndicatorSource] = frozenset(
            source for item in selected for source in _condition_sources(item)
        )
        if covered != allowed:
            missing = sorted(_SOURCE_KO[s] for s in (allowed - covered))
            skipped.append(
                f"조건 제거 '{label}': 원 전략에 {'·'.join(missing)} 조건이 없어 구성 불가"
            )
            continue

        variant_spec = base_spec.model_copy(update={"entry": ConditionGroup(all=list(selected))})
        try:
            prepared = _prepare_signals(variant_spec, joined_full, start_date, end_date)
        except StrategyValidationError as err:
            skipped.append(f"조건 제거 '{label}': 변형 전략 컴파일 실패 — {err}")
            continue

        result = _run_engine(
            prepared, base_config, benchmark_series, start_date, end_date, fs_scope
        )
        # 전체 조건 변형은 원 전략과 동일 — 연구용 경로가 원 전략을 그대로 재현하는지 자기 검증.
        if allowed == present_sources:
            _assert_matches_base(label, result, base_result)

        results.append(
            AblationResult(
                variant=label,
                sources=[s.value for s in sorted(allowed)],
                num_conditions=len(selected),
                num_trades=result.num_trades,
                cumulative_return=result.cumulative_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
            )
        )
    return results


def _assert_matches_base(label: str, result: BacktestResult, base_result: BacktestResult) -> None:
    """전체 조건 변형이 원 전략(base) 결과와 일치하는지 확인한다(자기 검증, 명세 W3c §2.1)."""
    if result.num_trades != base_result.num_trades or not _close(
        result.cumulative_return, base_result.cumulative_return
    ):
        raise AssertionError(
            f"조건 제거 '{label}'(전체 조건)이 원 전략과 불일치 — 연구용 경로 재현 오류: "
            f"num_trades {result.num_trades} vs {base_result.num_trades}, "
            f"cumulative_return {result.cumulative_return} vs {base_result.cumulative_return}"
        )


# --- 비용 민감도 (§24.2) -----------------------------------------------------


def _run_cost_sensitivity(
    prepared: _PreparedSignals,
    base_config: BacktestConfig,
    benchmark_series: pd.Series | None,
    start_date: date,
    end_date: date,
    fs_scope: str,
) -> list[CostSensitivityResult]:
    """commission·sell_tax·slippage를 0배/1배/2배 배율로 원 전략을 재실행한다(§24.2)."""
    results: list[CostSensitivityResult] = []
    for multiplier in _COST_MULTIPLIERS:
        config = base_config.model_copy(
            update={
                "commission_rate": base_config.commission_rate * multiplier,
                "sell_tax_rate": base_config.sell_tax_rate * multiplier,
                "slippage_rate": base_config.slippage_rate * multiplier,
            }
        )
        result = _run_engine(prepared, config, benchmark_series, start_date, end_date, fs_scope)
        results.append(
            CostSensitivityResult(
                multiplier=multiplier,
                commission_rate=config.commission_rate,
                sell_tax_rate=config.sell_tax_rate,
                slippage_rate=config.slippage_rate,
                num_trades=result.num_trades,
                cumulative_return=result.cumulative_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
            )
        )
    return results


# --- 하위 기간 (§24.2) -------------------------------------------------------


def _run_subperiods(
    base_spec: StrategySpec,
    base_frame: pd.DataFrame,
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    base_config: BacktestConfig,
    benchmark_series: pd.Series | None,
    start_date: date,
    end_date: date,
    fs_scope: str,
    skipped: list[str],
) -> list[SubperiodResult]:
    """[start, mid]·[mid, end] 이분할(mid=거래일 기준 중앙)로 원 전략을 재실행한다(§24.2)."""
    trading_days: list[date] = list(base_frame.index)
    if len(trading_days) < 2:
        skipped.append("하위 기간 분석: [start, end] 거래일이 2일 미만이라 이분할 불가")
        return []

    mid = trading_days[len(trading_days) // 2]
    windows = [("전반부", start_date, mid), ("후반부", mid, end_date)]

    results: list[SubperiodResult] = []
    for label, window_start, window_end in windows:
        joined = build_backtest_frame(
            daily, metrics, fs_scope=fs_scope, start_date=window_start, end_date=window_end
        )
        assert_no_lookahead(joined)
        prepared = _prepare_signals(base_spec, joined, window_start, window_end)
        result = _run_engine(
            prepared, base_config, benchmark_series, window_start, window_end, fs_scope
        )
        results.append(
            SubperiodResult(
                label=label,
                start_date=window_start,
                end_date=window_end,
                num_trades=result.num_trades,
                cumulative_return=result.cumulative_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
            )
        )
    return results


# --- 파이프라인 구성 요소 ------------------------------------------------------


def _prepare_signals(
    spec: StrategySpec, joined: pd.DataFrame, start_date: date, end_date: date
) -> _PreparedSignals:
    """전략을 컴파일해 지표·진입·청산 신호를 계산하고 [start, end]로 절단한다.

    ``execute_approved_strategy``와 동일한 순서(indicators → 신호 → 절단)를 따른다 —
    승인 백테스트와 결과가 일치하도록 하기 위함이다(명세 W3c §2.1). 컴파일 실패는
    :class:`StrategyValidationError`로 전파한다(호출부가 skipped에 기록).
    """
    compiled = compile_strategy(spec)
    with_indicators = compute_indicators(joined, compiled.required_columns)
    entry = entry_signal(compiled, with_indicators)
    exit_ = exit_signal(compiled, with_indicators)

    frame = truncate_to_window(with_indicators, start_date, end_date)
    return _PreparedSignals(
        frame=frame,
        entry=entry.reindex(frame.index),
        exit_=exit_.reindex(frame.index),
        position_rules=compiled.position_rules,
        strategy_name=spec.strategy_name,
    )


def _run_engine(
    prepared: _PreparedSignals,
    config: BacktestConfig,
    benchmark_series: pd.Series | None,
    start_date: date,
    end_date: date,
    fs_scope: str,
) -> BacktestResult:
    """준비된 신호로 체결 시뮬레이션·성과지표를 계산한다(연구용 경로, 게이트 우회)."""
    engine_result = run_backtest(
        prepared.frame, prepared.entry, prepared.exit_, prepared.position_rules, config
    )
    benchmark_close = _benchmark_for_index(benchmark_series, prepared.frame.index)
    return compute_backtest_metrics(
        engine_result=engine_result,
        asset_frame=prepared.frame,
        benchmark_close=benchmark_close,
        config=config,
        strategy_name=prepared.strategy_name,
        start_date=start_date,
        end_date=end_date,
        fs_scope=fs_scope,
    )


# --- 조건 소스 분류 (§24.3) --------------------------------------------------


def _entry_items(spec: StrategySpec) -> list[Condition | ConditionGroup]:
    """``entry.all``의 최상위 조건 목록을 꺼낸다 (entry는 all 그룹, 명세 A5 §2)."""
    return list(spec.entry.all) if spec.entry.all is not None else []


def _condition_sources(node: Condition | ConditionGroup) -> frozenset[IndicatorSource]:
    """조건(그룹)이 참조하는 지표들의 소스 집합을 구한다 (registry.resolve_indicator)."""
    return frozenset(resolve_indicator(name).source for name in _collect_indicator_names(node))


def _collect_indicator_names(node: Condition | ConditionGroup) -> set[str]:
    """조건 트리(all/any 재귀)에서 등장하는 모든 지표명(left + 컬럼 참조 right)을 모은다."""
    if isinstance(node, Condition):
        names = {node.left}
        if isinstance(node.right, str):
            names.add(node.right)
        return names
    items = node.all if node.all is not None else node.any
    collected: set[str] = set()
    for item in items or []:
        collected |= _collect_indicator_names(item)
    return collected


# --- 데이터 로드 (runner 관례와 동일 경로) ------------------------------------


def _load_daily(data_dir: Path, stock_code: str) -> pd.DataFrame:
    """A3 정규화 종목 daily.parquet을 로드한다(runner._load_daily와 동일 경로)."""
    path = data_dir / "normalized" / "market" / stock_code / "daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"종목 daily.parquet이 없습니다: {path} (r2b collect-market으로 먼저 수집)"
        )
    return pd.read_parquet(path)


def _load_metrics(data_dir: Path, corp_code: str) -> pd.DataFrame:
    """A4 financial_metrics.parquet을 로드한다(runner._load_metrics와 동일 경로)."""
    path = data_dir / "normalized" / "financials" / corp_code / "financial_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"financial_metrics.parquet이 없습니다: {path} (r2b build-financials로 먼저 생성)"
        )
    return pd.read_parquet(path)


def _load_benchmark_series(data_dir: Path, benchmark: str) -> pd.Series | None:
    """벤치마크 지수 종가 Series(index=date)를 로드한다 — 데이터 없으면 None(NaN 처리)."""
    index_code = _BENCHMARK_INDEX_CODES.get(benchmark, benchmark)
    path = data_dir / "normalized" / "market" / f"index_{index_code}" / "daily.parquet"
    if not path.exists():
        logger.warning("벤치마크 지수 데이터 없음: %s — 강건성 벤치마크 지표는 None 처리", path)
        return None
    frame = pd.read_parquet(path)
    dates = [as_date(value) for value in frame["date"]]
    return pd.Series(
        frame["close"].to_numpy(dtype="float64"),
        index=pd.Index(dates, name="date"),
        name="benchmark_close",
    )


def _benchmark_for_index(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """벤치마크 Series를 대상 거래일 index에 정렬한다 — 없으면 전 구간 NaN."""
    if series is None:
        return pd.Series([float("nan")] * len(index), index=index, name="benchmark_close")
    reindexed: pd.Series = series.reindex(index)
    reindexed.name = "benchmark_close"
    return reindexed


def _close(a: float | None, b: float | None, *, tol: float = 1e-9) -> bool:
    """두 지표값이 부동소수 오차 범위에서 같은지(둘 다 None도 같음) 판정한다."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


__all__ = [
    "AblationResult",
    "CostSensitivityResult",
    "RobustnessReport",
    "SubperiodResult",
    "run_robustness",
]
