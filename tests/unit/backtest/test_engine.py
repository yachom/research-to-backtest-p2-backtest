"""engine.py — 체결 시뮬레이션 손계산 대조 (명세 A6 §3·§6, README §23.3·§28.3).

§28.3-3(t 종가 신호가 t 종가·t 시가에 체결되면 실패 — 체결은 반드시 t+1 open)을
포함한다. 진입·청산 각 사유(SIGNAL/MAX_HOLDING/STOP_LOSS/END_OF_DATA), 정수
주식수·현금 잔여, 재진입 시퀀스, 비용 반영을 손으로 계산해 대조한다.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
import pytest

from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.backtest.engine import ExitReason, run_backtest
from research_backtest.quant.strategy.compiler import PositionRules

DailyFactory = Callable[..., pd.DataFrame]
SignalFactory = Callable[[pd.Index, list[date]], pd.Series]
ConfigFactory = Callable[..., BacktestConfig]

NO_RULES = PositionRules(max_holding_days=None, stop_loss=None)
FROM = date(2024, 1, 1)


def _dates(frame: pd.DataFrame) -> list[date]:
    return list(frame.index)


# --- §28.3-3: 체결은 반드시 t+1 시가 ------------------------------------------


def test_entry_fills_at_next_open_not_same_close_or_open(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """§28.3-3: t 종가 신호는 t+1 시가에 체결된다(t 종가·t 시가 아님)."""
    daily = make_daily(opens=[100, 110, 120, 130], closes=[105, 115, 125, 135], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[1]])  # 신호일 = index1
    exit_ = make_signal(daily.index, [])
    result = run_backtest(daily, entry, exit_, NO_RULES, bt_config())

    trade = result.trades[0]
    assert trade.entry_signal_date == d[1]
    assert trade.entry_date == d[2]  # t+1
    assert trade.entry_price == 120  # open[2]
    assert trade.entry_price != 115  # t 종가 아님
    assert trade.entry_price != 110  # t 시가 아님


def test_last_day_signal_is_discarded(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """마지막 거래일 신호는 체결 불가로 폐기된다(t+1이 없음)."""
    daily = make_daily(opens=[100, 100, 100], closes=[100, 100, 100], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[2]])  # 마지막 날 신호
    result = run_backtest(daily, entry, make_signal(daily.index, []), NO_RULES, bt_config())
    assert result.trades == []
    assert all(row.position == 0 for row in result.daily)


# --- 정수 주식수·현금 잔여 ----------------------------------------------------


def test_integer_shares_and_residual_cash(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """주식 수 = floor(cash/체결가), 잔여는 현금(비용 0)."""
    daily = make_daily(opens=[100, 120, 120, 120], closes=[100, 120, 120, 120], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 신호 index0 → 체결 index1 open=120
    result = run_backtest(daily, entry, make_signal(daily.index, []), NO_RULES, bt_config())

    # shares = floor(1_000_000/120) = 8333, 잔여 = 1_000_000 - 8333*120 = 40
    entry_row = result.daily[1]
    assert entry_row.shares == 8333
    assert entry_row.cash == pytest.approx(40.0)
    assert entry_row.equity == pytest.approx(40.0 + 8333 * 120)


# --- 청산 사유별 손계산 -------------------------------------------------------


def test_exit_signal_pnl(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """조건 exit(SIGNAL) — 진입·청산 손계산."""
    daily = make_daily(opens=[10, 20, 30, 40, 50], closes=[10, 20, 30, 40, 50], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 체결 index1 open=20
    exit_ = make_signal(daily.index, [d[2]])  # 청산 신호 index2 → 체결 index3 open=40
    result = run_backtest(daily, entry, exit_, NO_RULES, bt_config())

    trade = result.trades[0]
    assert trade.exit_reason == ExitReason.SIGNAL
    assert trade.entry_price == 20 and trade.exit_price == 40
    assert trade.shares == 50_000  # floor(1e6/20)
    assert trade.holding_days == 2  # exit_index(3) - entry_index(1)
    assert trade.pnl == pytest.approx(1_000_000.0)  # 50000*(40-20)
    assert trade.pnl_pct == pytest.approx(1.0)
    assert trade.exit_date == d[3]


def test_stop_loss_priority_on_close(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """stop_loss — 종가/진입가-1 <= -0.10에서 발동, t+1 시가 체결."""
    daily = make_daily(opens=[100, 100, 100, 90], closes=[100, 100, 85, 90], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 체결 index1 open=100 → entry_price 100
    rules = PositionRules(max_holding_days=None, stop_loss=-0.10)
    result = run_backtest(daily, entry, make_signal(daily.index, []), rules, bt_config())

    trade = result.trades[0]
    # index2 종가 85: 85/100-1 = -0.15 <= -0.10 → STOP_LOSS 신호 → index3 시가 90 체결
    assert trade.exit_reason == ExitReason.STOP_LOSS
    assert trade.exit_date == d[3] and trade.exit_price == 90
    assert trade.shares == 10_000
    assert trade.pnl == pytest.approx(-100_000.0)  # 10000*(90-100)


def test_max_holding_days(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """max_holding_days — 보유 거래일(진입 체결일 포함) >= N에서 청산."""
    daily = make_daily(opens=[100] * 6, closes=[100] * 6, start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 체결 index1(entry_index=1, 보유 1일차)
    rules = PositionRules(max_holding_days=3, stop_loss=None)
    result = run_backtest(daily, entry, make_signal(daily.index, []), rules, bt_config())

    trade = result.trades[0]
    # index3: 보유일수 = 3-1+1 = 3 >= 3 → 신호 → index4 시가 체결
    assert trade.exit_reason == ExitReason.MAX_HOLDING
    assert trade.exit_date == d[4]
    assert trade.holding_days == 3  # exit_index(4) - entry_index(1)


def test_end_of_data_forced_close(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """데이터 종료 시 미청산 포지션은 마지막 종가로 강제 청산(END_OF_DATA)."""
    daily = make_daily(opens=[100, 100, 100], closes=[100, 100, 130], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 체결 index1, 이후 청산 신호 없음
    result = run_backtest(daily, entry, make_signal(daily.index, []), NO_RULES, bt_config())

    trade = result.trades[0]
    assert trade.exit_reason == ExitReason.END_OF_DATA
    assert trade.exit_date == d[2]  # 마지막 날
    assert trade.exit_price == 130  # 마지막 종가(시가 아님)
    assert result.daily[-1].position == 0  # 강제 청산 후 flat


def test_reentry_sequence_no_same_bar(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """청산 체결일 종가 신호로 재진입 판정 → t+2 시가 진입(같은 봉 재진입 없음)."""
    daily = make_daily(opens=[10, 20, 30, 40, 50], closes=[10, 20, 30, 40, 50], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0], d[2]])  # 진입 신호 index0, index2
    exit_ = make_signal(daily.index, [d[1]])  # 청산 신호 index1 → 매도 index2 시가
    result = run_backtest(daily, entry, exit_, NO_RULES, bt_config())

    assert len(result.trades) == 2
    first, second = result.trades
    assert first.entry_date == d[1] and first.exit_date == d[2]  # 매도 index2
    # index2 종가에서 재진입 판정 → index3 시가 진입(같은 봉 index2 아님)
    assert second.entry_signal_date == d[2]
    assert second.entry_date == d[3]
    assert second.exit_reason == ExitReason.END_OF_DATA


# --- 비용 반영 손계산 ---------------------------------------------------------


def test_costs_applied_to_entry_and_exit(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """수수료(양편)·슬리피지·매도세 손계산 대조.

    잔여현금 < 진입수수료면 cash가 근소 음수가 될 수 있다(스펙: 주식 수 =
    floor(cash/체결가) 전량 매수) — 공식 그대로 검증한다.
    """
    daily = make_daily(opens=[100, 100, 100, 200], closes=[100, 100, 100, 200], start=FROM)
    d = _dates(daily)
    entry = make_signal(daily.index, [d[0]])  # 체결 index1
    exit_ = make_signal(daily.index, [d[2]])  # 청산 신호 index2 → 체결 index3
    config = bt_config(commission_rate=0.001, sell_tax_rate=0.002, slippage_rate=0.01)
    result = run_backtest(daily, entry, exit_, NO_RULES, config)

    trade = result.trades[0]
    # 진입: fill = 100*1.01 = 101, shares = floor(1e6/101) = 9900
    assert trade.entry_price == pytest.approx(101.0)
    assert trade.shares == 9900
    entry_commission = 9900 * 101 * 0.001  # 999.9
    # 청산: fill = 200*0.99 = 198, gross = 9900*198 = 1_960_200
    assert trade.exit_price == pytest.approx(198.0)
    gross_sell = 9900 * 198
    exit_commission = gross_sell * 0.001
    exit_tax = gross_sell * 0.002
    proceeds = gross_sell - exit_commission - exit_tax
    cost_basis = 9900 * 101 + entry_commission
    assert trade.costs == pytest.approx(entry_commission + exit_commission + exit_tax)
    assert trade.pnl == pytest.approx(proceeds - cost_basis)
    assert trade.pnl_pct == pytest.approx((proceeds - cost_basis) / cost_basis)


def test_no_signals_stays_flat(
    make_daily: DailyFactory, make_signal: SignalFactory, bt_config: ConfigFactory
) -> None:
    """신호가 없으면 무포지션·무거래(초기자본 유지)."""
    daily = make_daily(opens=[100] * 4, closes=[100] * 4, start=FROM)
    result = run_backtest(
        daily, make_signal(daily.index, []), make_signal(daily.index, []), NO_RULES, bt_config()
    )
    assert result.trades == []
    assert all(row.equity == pytest.approx(1_000_000.0) for row in result.daily)
    assert all(row.daily_return == 0.0 for row in result.daily)
