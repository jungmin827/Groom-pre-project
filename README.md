# KorQuAD RAG 시스템

한국어 질의응답을 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 아키텍처

```
korquad-rag/
├── src/app/
│   ├── main.py          # FastAPI 애플리케이션 진입점
│   ├── api/routes.py    # API 라우트 정의
│   ├── core/config.py   # 설정 관리
│   ├── data/preprocess.py # 데이터 전처리
│   ├── embeddings/      # 임베딩 생성
│   ├── retriever/       # 문서 검색
│   └── models/          # LLM 모델 래퍼
├── tests/               # 테스트 파일
└── scripts/            # 유틸리티 스크립트
```

## 설치 및 실행

### 1. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 테스트 실행
```bash
pytest -q
```

### 4. 개발 서버 실행
```bash
uvicorn src.app.main:app --reload --port 8000
```

## API 사용법

### 헬스 체크
```bash
curl http://localhost:8000/health
```

### 질의응답
```bash
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "한국의 수도는 어디인가요?",
       "context": "서울은 대한민국의 수도입니다."
     }'
```

## 개발

개발 환경 설정을 위한 스크립트를 사용할 수 있습니다:

```bash
chmod +x scripts/run_dev.sh
./scripts/run_dev.sh
```

## Docker 실행

```bash
docker build -t korquad-rag .
docker run -p 8000:8000 korquad-rag
```
