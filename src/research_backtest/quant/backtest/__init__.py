"""백테스트 엔진 — 신호→체결 시뮬레이션·성과지표·게이트 강제 진입점 (명세 A6, README §22~§24).

전략 DSL(A5)이 만든 신호를 과거 데이터로 검증한다. 정확성의 핵심은
**룩어헤드 방지**다(README §22.1): 재무 수치는 as-of join으로 available_from
이후에만 보이고, t일 종가 신호는 t+1 거래일 시가에만 체결된다(§23.3).

- data: daily(A3) + financial_metrics(A4) as-of join + assert_no_lookahead (명세 A6 §2)
- engine: 포지션 상태 머신·체결 규칙·TradeRecord·DailyPortfolioRow (명세 A6 §3)
- costs: configs/backtest.yaml 로더 + BacktestConfig (명세 A6 §5 소비)
- metrics: 성과지표(§24.1)·엣지 None 처리·B&H 비교 (명세 A6 §4)
- runner: gates.ensure_strategy_approved 강제 진입점 + 산출물 3종 저장 (명세 A6 §5)
"""
