"""체결 시뮬레이션 — 포지션 상태 머신 (명세 A6 §3, README §23.3).

single asset · long/cash. 신호는 t일 **종가 기준**으로 평가하고(entry/exit
Series는 A5가 t까지 정보만으로 계산), 체결은 **t+1 거래일 시가**다 — 종가 신호를
당일에 체결하지 않는 것이 룩어헤드 방지의 핵심이다(README §22.1·§28.3).

체결 규칙(명세 A6 §3):

- 진입: flat ∧ entry[t] → t+1 시가 매수. 체결가 = ``open*(1+slippage)``,
  수수료 = 체결금액*commission_rate, 주식 수 = ``floor(cash/체결가)`` 정수.
- 청산(t 종가 판정, 우선순위): ① stop_loss ``close/entry_price-1 <= stop``
  (진입가 대비 종가 — 장중 저가 미반영, 아래 한계 참조) ② max_holding_days
  보유 거래일(진입 체결일 포함) ≥ N ③ 조건 exit[t] → t+1 시가 매도.
  체결가 = ``open*(1-slippage)``, 수수료 + ``sell_tax``(매도 금액 기준).
- 동시 발생 시 청산 우선. 청산 체결일(t+1) 종가 신호로 재진입 판정 가능(→ t+2
  시가) — 같은 봉 재진입은 없다.
- 마지막 거래일 신호는 체결 불가로 폐기한다. 데이터 종료 시 미청산 포지션은
  마지막 **종가**로 강제 청산하고 ``exit_reason=END_OF_DATA``로 표기한다.

**stop_loss 한계**: 종가 기준으로만 판정하므로 장중 저가가 손절선을 하회했다가
종가가 회복하면 손절이 발동하지 않는다 — 실거래보다 손절이 늦거나 누락될 수
있다(보수적이지 않은 방향). MVP 설계 결정이다.

engine을 직접 호출하는 것은 테스트·연구용이다 — 실행의 공식 진입점은
:func:`runner.execute_approved_strategy`(승인 게이트 강제)다(명세 A6 §5).
"""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict

from research_backtest.quant.backtest.costs import BacktestConfig
from research_backtest.quant.strategy.compiler import PositionRules


class ExitReason(StrEnum):
    """청산 사유 (명세 A6 §3, TradeRecord.exit_reason 허용값)."""

    SIGNAL = "SIGNAL"  # 조건 기반 exit_signal
    MAX_HOLDING = "MAX_HOLDING"  # 최대 보유기간 도달
    STOP_LOSS = "STOP_LOSS"  # 진입가 대비 손절
    END_OF_DATA = "END_OF_DATA"  # 데이터 종료 시 강제 청산


class TradeRecord(BaseModel):
    """왕복 거래 1건 (명세 A6 §3).

    - ``entry_price``/``exit_price``: 슬리피지가 반영된 실제 체결가(원/주).
    - ``holding_days``: 진입 체결일부터 청산 체결일까지의 거래일 수(= 진입가로
      포지션을 종가에 보유한 거래일 수). 정상 청산은 청산 체결일 시가에
      팔았으므로 그날 종가 보유는 아니다.
    - ``pnl``: 왕복 순손익(원) = (매도 순수취금) - (매수 총지출). ``pnl_pct``는
      ``pnl / 매수총지출``. ``costs``는 진입·청산 수수료와 매도세 합계.
    """

    model_config = ConfigDict(extra="forbid")

    entry_signal_date: date
    entry_date: date
    entry_price: float
    shares: int

    exit_signal_date: date
    exit_date: date
    exit_price: float

    holding_days: int
    pnl: float
    pnl_pct: float
    exit_reason: ExitReason
    costs: float


class DailyPortfolioRow(BaseModel):
    """일별 포트폴리오 스냅샷 (명세 A6 §3).

    ``equity = cash + shares*close``. ``daily_return``은 전일 대비 equity 수익률
    (첫날 0.0). ``position``은 그날 종가 시점의 보유 여부(0/1)다.
    """

    model_config = ConfigDict(extra="forbid")

    date: date
    position: int
    shares: int
    cash: float
    equity: float
    daily_return: float


class EngineResult(BaseModel):
    """엔진 실행 산출물 — 거래 목록 + 일별 포트폴리오 (명세 A6 §3)."""

    model_config = ConfigDict(extra="forbid")

    trades: list[TradeRecord]
    daily: list[DailyPortfolioRow]

    def trade_frame(self) -> pd.DataFrame:
        """trade_log.csv용 DataFrame (명세 A6 §5)."""
        columns = list(TradeRecord.model_fields)
        records = [t.model_dump() for t in self.trades]
        return pd.DataFrame(records, columns=columns)

    def daily_frame(self) -> pd.DataFrame:
        """daily_portfolio.csv용 DataFrame (명세 A6 §5)."""
        columns = list(DailyPortfolioRow.model_fields)
        records = [row.model_dump() for row in self.daily]
        return pd.DataFrame(records, columns=columns)

    def equity_series(self) -> pd.Series:
        """equity 시계열(index=date) — metrics 계산 입력."""
        index = pd.Index([row.date for row in self.daily], name="date")
        return pd.Series([row.equity for row in self.daily], index=index, name="equity")


class _OpenPosition:
    """보유 중 포지션의 진입 상태 — 청산 시 TradeRecord를 만드는 데 필요한 정보."""

    __slots__ = (
        "entry_commission",
        "entry_date",
        "entry_index",
        "entry_price",
        "entry_signal_date",
        "gross_buy",
        "shares",
    )

    def __init__(
        self,
        *,
        entry_signal_date: date,
        entry_date: date,
        entry_index: int,
        entry_price: float,
        shares: int,
        gross_buy: float,
        entry_commission: float,
    ) -> None:
        self.entry_signal_date = entry_signal_date
        self.entry_date = entry_date
        self.entry_index = entry_index
        self.entry_price = entry_price
        self.shares = shares
        self.gross_buy = gross_buy
        self.entry_commission = entry_commission


def run_backtest(
    frame: pd.DataFrame,
    entry: pd.Series,
    exit_: pd.Series,
    position_rules: PositionRules,
    config: BacktestConfig,
) -> EngineResult:
    """신호를 체결 시뮬레이션해 거래·일별 포트폴리오를 만든다 (명세 A6 §3).

    ``frame``은 ``open``·``close``를 포함하고 ``date``(datetime.date) 오름차순
    index를 가진 [start, end] 절단 프레임이다(:func:`data.truncate_to_window`
    이후). ``entry``/``exit_``는 A5 ``entry_signal``/``exit_signal`` 산출
    bool Series이며 frame.index에 정렬한다. NaN·미정렬은 False로 처리한다.
    """
    if not frame.index.is_monotonic_increasing:
        raise ValueError(
            "frame이 거래일 오름차순으로 정렬되어 있지 않습니다 (data.py 절단 결과 사용)."
        )

    dates: list[date] = list(frame.index)
    n = len(dates)
    opens = frame["open"].astype("float64").to_numpy()
    closes = frame["close"].astype("float64").to_numpy()
    entry_arr = entry.reindex(frame.index).fillna(False).astype(bool).to_numpy()
    exit_arr = exit_.reindex(frame.index).fillna(False).astype(bool).to_numpy()

    slippage = config.slippage_rate
    commission_rate = config.commission_rate
    sell_tax_rate = config.sell_tax_rate

    cash = float(config.initial_cash)
    open_position: _OpenPosition | None = None
    # 전일 종가에 예약된 주문: ("BUY", signal_date) | ("SELL", signal_date, ExitReason)
    pending: tuple[str, date] | tuple[str, date, ExitReason] | None = None

    trades: list[TradeRecord] = []
    equities: list[float] = []
    positions: list[int] = []
    shares_held: list[int] = []
    cashes: list[float] = []

    for i in range(n):
        open_i = opens[i]
        close_i = closes[i]
        is_last = i == n - 1

        # (A) 전일 종가에 예약된 주문을 오늘 시가에 체결
        if pending is not None:
            if pending[0] == "BUY":
                fill_price = open_i * (1.0 + slippage)
                n_shares = math.floor(cash / fill_price) if fill_price > 0 else 0
                if n_shares > 0:
                    gross_buy = n_shares * fill_price
                    commission = gross_buy * commission_rate
                    cash -= gross_buy + commission
                    open_position = _OpenPosition(
                        entry_signal_date=pending[1],
                        entry_date=dates[i],
                        entry_index=i,
                        entry_price=fill_price,
                        shares=n_shares,
                        gross_buy=gross_buy,
                        entry_commission=commission,
                    )
                # n_shares == 0(현금 부족)이면 진입 실패 — flat 유지
            else:  # SELL
                assert open_position is not None
                exit_reason = pending[2]  # type: ignore[misc]
                fill_price = open_i * (1.0 - slippage)
                cash, trade = _close_position(
                    open_position,
                    exit_signal_date=pending[1],
                    exit_date=dates[i],
                    exit_index=i,
                    exit_price=fill_price,
                    exit_reason=exit_reason,
                    cash=cash,
                    commission_rate=commission_rate,
                    sell_tax_rate=sell_tax_rate,
                )
                trades.append(trade)
                open_position = None
            pending = None

        # (B) 오늘 종가 시점 판정 → 내일 체결 예약(또는 마지막 날 강제 청산)
        if open_position is not None:
            if is_last:
                # 데이터 종료 — 마지막 종가로 강제 청산(슬리피지 없음), END_OF_DATA
                cash, trade = _close_position(
                    open_position,
                    exit_signal_date=dates[i],
                    exit_date=dates[i],
                    exit_index=i,
                    exit_price=close_i,
                    exit_reason=ExitReason.END_OF_DATA,
                    cash=cash,
                    commission_rate=commission_rate,
                    sell_tax_rate=sell_tax_rate,
                )
                trades.append(trade)
                open_position = None
            else:
                reason = _exit_reason(open_position, i, close_i, exit_arr[i], position_rules)
                if reason is not None:
                    pending = ("SELL", dates[i], reason)
        elif entry_arr[i] and not is_last:
            pending = ("BUY", dates[i])

        # (C) 오늘 종가 시점 일별 스냅샷
        pos_shares = open_position.shares if open_position is not None else 0
        equity = cash + pos_shares * close_i
        positions.append(1 if open_position is not None else 0)
        shares_held.append(pos_shares)
        cashes.append(cash)
        equities.append(equity)

    daily_returns = _daily_returns(equities)
    daily = [
        DailyPortfolioRow(
            date=dates[i],
            position=positions[i],
            shares=shares_held[i],
            cash=cashes[i],
            equity=equities[i],
            daily_return=daily_returns[i],
        )
        for i in range(n)
    ]
    return EngineResult(trades=trades, daily=daily)


# --- 내부 구현 ---------------------------------------------------------------


def _exit_reason(
    position: _OpenPosition,
    i: int,
    close_i: float,
    exit_signal_i: bool,
    rules: PositionRules,
) -> ExitReason | None:
    """t 종가 시점 청산 사유를 우선순위(stop_loss > max_holding > signal)로 판정한다."""
    if rules.stop_loss is not None and (close_i / position.entry_price - 1.0) <= rules.stop_loss:
        return ExitReason.STOP_LOSS
    # 보유 거래일 수 — 진입 체결일 포함(명세 A6 §3)
    holding_incl = i - position.entry_index + 1
    if rules.max_holding_days is not None and holding_incl >= rules.max_holding_days:
        return ExitReason.MAX_HOLDING
    if exit_signal_i:
        return ExitReason.SIGNAL
    return None


def _close_position(
    position: _OpenPosition,
    *,
    exit_signal_date: date,
    exit_date: date,
    exit_index: int,
    exit_price: float,
    exit_reason: ExitReason,
    cash: float,
    commission_rate: float,
    sell_tax_rate: float,
) -> tuple[float, TradeRecord]:
    """포지션을 청산해 (갱신된 cash, TradeRecord)를 만든다.

    매도 금액에 수수료와 매도세를 부과한다. holding_days는 진입·청산 체결
    index 차(= 종가에 보유한 거래일 수)다.
    """
    gross_sell = position.shares * exit_price
    exit_commission = gross_sell * commission_rate
    exit_tax = gross_sell * sell_tax_rate
    proceeds = gross_sell - exit_commission - exit_tax
    cash += proceeds

    cost_basis = position.gross_buy + position.entry_commission
    pnl = proceeds - cost_basis
    pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0
    total_costs = position.entry_commission + exit_commission + exit_tax

    trade = TradeRecord(
        entry_signal_date=position.entry_signal_date,
        entry_date=position.entry_date,
        entry_price=position.entry_price,
        shares=position.shares,
        exit_signal_date=exit_signal_date,
        exit_date=exit_date,
        exit_price=exit_price,
        holding_days=exit_index - position.entry_index,
        pnl=pnl,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
        costs=total_costs,
    )
    return cash, trade


def _daily_returns(equities: list[float]) -> list[float]:
    """equity 시계열의 전일 대비 수익률(첫날 0.0)."""
    returns = [0.0]
    for i in range(1, len(equities)):
        prev = equities[i - 1]
        returns.append(equities[i] / prev - 1.0 if prev != 0 else 0.0)
    return returns
