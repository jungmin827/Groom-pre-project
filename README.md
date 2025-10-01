<<<<<<< HEAD
# Groom-pre-project
구름의 사전과제를 위한 레포지토리입니다
=======
# KorQuAD RAG System

한국어 질의응답을 위한 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 아키텍처

```
src/
├── app/
│   ├── main.py              # FastAPI 애플리케이션 진입점
│   ├── api/
│   │   └── routes.py        # API 라우터 (/health, /qa)
│   ├── core/
│   │   └── config.py        # 설정 관리 (Pydantic Settings)
│   ├── data/
│   │   └── preprocess.py    # 데이터 전처리 (KorQuAD 파싱, 청킹)
│   ├── embeddings/
│   │   └── embeddings.py    # 임베딩 생성 및 관리
│   ├── retriever/
│   │   └── retriever.py     # 문서 검색 및 관련성 스코어링
│   └── models/
│       └── llm.py          # LLM 래퍼 (모델 로딩, 생성)
├── tests/
│   └── test_smoke.py       # 기본 테스트
└── scripts/
    └── run_dev.sh          # 개발 환경 실행 스크립트
```

## 설치 및 실행

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는 .venv\Scripts\activate  # Windows
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

## API 사용 예시

### 헬스 체크

```bash
curl http://localhost:8000/health
```

응답:
```json
{"status":"ok"}
```

### 질의응답

```bash
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "대한민국의 수도는 어디인가요?",
       "context": "대한민국은 동아시아에 위치한 나라입니다."
     }'
```

응답:
```json
{
  "answer": "not implemented yet",
  "sources": []
}
```

## 환경 변수

- `CHROMA_DIR`: ChromaDB 저장 디렉토리 (기본값: ./chroma_db)
- `EMBEDDING_MODEL`: 임베딩 모델명 (기본값: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- `LLM_MODEL`: LLM 모델명 (기본값: microsoft/DialoGPT-medium)
- `PORT`: 서버 포트 (기본값: 8000)

## 개발

```bash
# 개발 스크립트 실행
./scripts/run_dev.sh

# 테스트만 실행
pytest

# 특정 테스트 실행
pytest tests/test_smoke.py -v
```
>>>>>>> 543202c (chore: scaffold project structure)
