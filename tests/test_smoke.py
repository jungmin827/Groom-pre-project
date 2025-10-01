"""기본 스모크 테스트 모듈"""

import pytest
import httpx
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture
def client():
    """테스트 클라이언트 픽스처"""
    return TestClient(app)


def test_health_endpoint(client):
    """헬스 체크 엔드포인트 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["message"] == "KorQuAD RAG API"


def test_qa_endpoint_stub(client):
    """질의응답 엔드포인트 스텁 테스트"""
    test_data = {
        "question": "대한민국의 수도는 어디인가요?",
        "context": "대한민국은 동아시아에 위치한 나라입니다."
    }
    
    response = client.post("/qa", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "not implemented yet"
    assert data["sources"] == []


def test_api_v1_health_endpoint(client):
    """API v1 헬스 체크 엔드포인트 테스트"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_v1_qa_endpoint_stub(client):
    """API v1 질의응답 엔드포인트 스텁 테스트"""
    test_data = {
        "question": "테스트 질문",
        "context": "테스트 컨텍스트"
    }
    
    response = client.post("/api/v1/qa", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == "not implemented yet"
    assert data["sources"] == []


def test_docs_endpoint(client):
    """API 문서 엔드포인트 테스트"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint(client):
    """ReDoc 문서 엔드포인트 테스트"""
    response = client.get("/redoc")
    assert response.status_code == 200


def test_invalid_qa_request(client):
    """잘못된 질의응답 요청 테스트"""
    # 필수 필드 누락
    response = client.post("/api/v1/qa", json={})
    assert response.status_code == 422  # Validation Error
    
    # 잘못된 데이터 타입
    response = client.post("/api/v1/qa", json={"question": 123})
    assert response.status_code == 422  # Validation Error
