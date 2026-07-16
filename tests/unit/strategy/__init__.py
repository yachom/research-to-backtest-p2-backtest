"""전략 DSL(quant.strategy) 단위 테스트 패키지 (명세 A5 §6).

``__init__.py``를 둔 이유: 이 디렉토리는 ``tests/unit/`` 자체에는
``__init__.py``가 없는 flat 레이아웃 위에서 pytest의 기본(prepend) 임포트
모드가 ``test_schema.py``처럼 흔한 모듈명을 다른 병렬 작업트리(A4/B1B2 등)의
동일 basename 테스트 파일과 충돌 없이 ``strategy.test_schema``로 한정해
수집하도록 하기 위함이다(병합 시 충돌 방지, MILESTONES D8 병렬 웨이브).
"""
