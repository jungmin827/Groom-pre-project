"""API 라우터 모듈"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any


class QARequest(BaseModel):
    """질의응답 요청 모델"""
    question: str
    context: Optional[str] = None


class QAResponse(BaseModel):
    """질의응답 응답 모델"""
    answer: str
    sources: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str


# 라우터 생성
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크 엔드포인트"""
    return HealthResponse(status="ok")


@router.post("/qa", response_model=QAResponse)
async def question_answer(request: QARequest):
    """질의응답 엔드포인트 (현재는 스텁 구현)"""
    # TODO: 실제 RAG 시스템 구현
    return QAResponse(
        answer="not implemented yet",
        sources=[]
    )
