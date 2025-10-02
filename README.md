# KorQuAD RAG 시스템

질의응답을 위한 RAG (Retrieval-Augmented Generation) 챗봇 서비스입니다.

---

### 1. 문제 정의 및 아키텍처 설계 🧠

이 프로젝트는 주어진 질문에 대해 정확하고 신뢰성 있는 답변을 제공하기 위해 RAG(검색 증강 생성) 시스템을 구현하는 것을 목표로 합니다. 아키텍처는 다음과 같이 역할과 책임을 명확하게 분리하여 모듈화되었습니다.

-   **`src/app/api/routes.py`**: FastAPI 라우터 모듈로, `health` 및 `qa` 엔드포인트를 정의하여 외부 요청을 처리합니다.
-   **`src/app/core/config.py`**: 애플리케이션의 설정(ChromaDB 디렉토리, 모델명 등)을 관리하여 코드와 설정을 분리합니다.
-   **`src/app/embeddings/embeddings.py`**: 텍스트를 임베딩 벡터로 변환하는 역할을 담당합니다.
-   **`src/app/retriever/retriever.py`**: 임베딩된 벡터를 기반으로 질문과 유사한 문서를 검색하고 관련성 점수를 계산하는 핵심 모듈입니다.
-   **`src/app/data/preprocess.py`**: 데이터 전처리를 담당하며, KorQuAD 데이터셋을 파싱하고 청킹(chunking)하여 모델이 활용하기 좋은 형태로 만듭니다.

각 모듈의 독립성을 높여 유지보수 및 확장이 용이하게 구조를 잡았습니다.

---

### 2. 데이터 전처리 🛠️

데이터는 `preprocess.py` 모듈에서 처리됩니다. 이 모듈은 KorQuAD 데이터셋을 효과적으로 활용하기 위해 불필요한 문장을 필터링하고, 텍스트를 청킹하는 역할을 수행합니다. `retriever.py`는 ChromaDB를 활용하여 전처리된 문서들을 벡터 데이터베이스에 저장하고 관리합니다.

이 시스템의 핵심 성능을 위해, 다음과 같은 기준에 따라 모델을 신중하게 선택했습니다.

- Embedding Model (Retriever): **intfloat/multilingual-e5-small**

- - **선택 이유**: 이 모델은 MTEB에서 높은 순위를 차지하며, 특히 다국어 환경에서 뛰어난 성능을 입증했습니다. 모델의 크기가 작아 로컬 환경에서도 빠르고 효율적으로 동작하면서도, 한국어 텍스트의 의미를 정확하게 벡터로 변환하는 능력이 매우 뛰어나 RAG 시스템의 첫 단계인 '검색(Retrieval)'에 가장 적합하다고 판단했습니다.

- LLM (Generator): **Qwen/Qwen2-1.5B-Instruct**

- - **선택 이유**: KorQuAD의 기반이 되는 위키피디아 데이터를 포함한 대규모 데이터로 학습하여, 검색된 컨텍스트에 대한 이해도가 매우 높습니다. 1.5B의 작은 파라미터 크기에도 불구하고, Fine-tuning된 'Instruct' 모델이기 때문에, "주어진 컨텍스트를 바탕으로 질문에 답변하라"는 RAG의 핵심 작업을 안정적으로 수행합니다.

---

### 3. 검색 정확도 🎯

-   `retriever.py`의 `search_similar` 메서드는 유사한 문서를 검색하며, 관련도 점수를 반환합니다.
-   `retrieve_for_qa` 메서드는 질문과 컨텍스트를 결합하여 검색 쿼리를 생성함으로써 검색의 정확도를 높입니다.
-   관련 없는 답변을 방지하기 위해 검색된 문서의 관련성 점수(유사도 거리)를 기반으로 **스코어 임계치(threshold)**를 설정하는 로직을 추가하여 품질을 관리할 예정입니다.

---

### 4. 응답 품질 ✨

- `qa` 엔드포인트는 답변과 함께 `sources` 리스트를 반환하도록 설계되었습니다.
- 답변의 출처를 명확히 제시하여 신뢰성을 확보하고, 사용자에게 더 나은 경험을 제공합니다.
- 관련 문서를 찾지 못한 경우 **fallback** 메시지를 제공하도록 구현되었습니다.
- 스트림릿 인터페이스를 통해 웹 환경에서 질문을 입력하고 답변을 받을 수 있습니다.

---

### 5. Fast api 구현 전략과 배포 전략 🤝

-   **설정 분리**: 모든 환경 변수는 `app/core/config.py`에서 관리됩니다.
-   **표준화된 API**: FastAPI를 사용하여 `/api/routes.py` 경로로 엔드포인트를 표준화하고, Pydantic 모델을 통해 JSON 스키마를 명확히 정의했습니다.
-   **CI/CD**: `.github/workflows/python-app.yml`에 테스트, 빌드, 배포를 위한 GitHub Actions CI/CD 파이프라인이 구성되어 있습니다.

---

### 설치 및 실행
중요: 모든 명령어는 프로젝트 최상위 폴더에서 실행하는 것을 기준으로 합니다
#### 1. 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```
#### 2. 의존성 설치
```Bash
pip install -r requirements.txt
```
#### 3. 개발 서버 실행
ChromaDB의 특정 버전에서 발생하는 Telemetry 관련 오류를 방지하기 위해 환경 변수를 설정한 후 서버를 실행합니다.
```Bash
export CHROMA_ANONYMIZED_TELEMETRY=false
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```
#### 4. 스트림릿 앱 실행
```Bash
streamlit run streamlit_chatbot.py --server.port 8501
```

#### API 사용 예시 (cURL)
Streamlit UI에서 편하게 질문을 통해 api 테스트를 할 수 있습니다!
요청 (Request)
터미널에서 아래 cURL 명령어를 사용하여 질의응답 API(api/v1/qa)에 POST 요청을 보낼 수 있습니다.
```
curl -X 'POST' \
  'http://localhost:8000/api/v1/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "임진왜란은 언제 발발했나요?",
  "top_k": 3
}'
```
응답 (Response)
요청이 성공하면 아래와 같은 형식의 JSON 응답을 받게 됩니다.
```
{
  "retrieved_document_id": "korquad_12345_chunk_0",
  "retrieved_document": "임진왜란은 1592년에 일본이 조선을 침략하면서 시작된 전쟁이다. ... (검색된 문서 내용)",
  "question": "임진왜란은 언제 발발했나요?",
  "answers": "1592년에 발발했습니다.",
  "quality_metrics": {
    "confidence": 0.85,
    "is_valid": true,
    "search_quality": {
      "total_results": 3,
      "avg_similarity_score": 0.91,
      "avg_relevance_score": 0.78
    }
  }
}
```
