"""전략 DSL — 스키마·지표 레지스트리·지표 계산·컴파일러 (README §20~§23, 명세 A5).

임의 Python 코드 실행 없이, 화이트리스트 지표·연산자만으로 선언적 전략
JSON을 검증·컴파일한다(README M8 DoD).

- schema: 전략 JSON pydantic 스키마 (README §23.4)
- registry: 허용 지표 화이트리스트 + lag(``_lag{n}``) 문법 (README §21)
- indicators: 가격·수급 지표 계산 — no-lookahead (README §21.2~§21.3)
- compiler: 검증 + entry_signal/exit_signal 신호 계산기 컴파일 (README §22~§23)
"""
