# KorQuAD RAG 시스템

한국어 질의응답을 위한 RAG (Retrieval-Augmented Generation) 시스템입니다.

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
