"""
API 라우트 정의
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.rag_chain import RAGChain
from app.core.config import settings
from app.core.data_loader import data_loader

router = APIRouter()

# RAG Chain 인스턴스 (싱글톤 패턴)
rag_chain = None

def get_rag_chain():
    """RAG Chain 인스턴스 반환 (비동기 로더 상태 확인)"""
    global rag_chain
    
    # 데이터 로더가 준비되지 않은 경우
    if not data_loader.is_ready:
        raise HTTPException(
            status_code=503, 
            detail={
                "message": "데이터 로딩 중입니다. 잠시 후 다시 시도해주세요.",
                "loading_status": data_loader.get_status()
            }
        )
    
    if rag_chain is None:
        try:
            rag_chain = RAGChain()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    return rag_chain

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QAResponse(BaseModel):
    retrieved_document_id: int
    retrieved_document: str
    question: str
    answers: str
    quality_metrics: Optional[Dict[str, Any]] = None

@router.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        rag = get_rag_chain()
        system_info = rag.get_system_info()
        return {
            "status": "ok", 
            "message": "서비스가 정상적으로 실행 중입니다",
            "system_info": system_info
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"시스템 초기화 오류: {str(e)}"
        }

@router.post("/qa", response_model=QAResponse)
async def question_answer(request: QARequest):
    """
    질의응답 엔드포인트
    
    Args:
        request: 질문과 검색할 문서 수를 포함하는 요청 객체
        
    Returns:
        답변과 참조 소스들을 포함하는 응답 객체
    """
    try:
        # RAG Chain 인스턴스 가져오기
        rag = get_rag_chain()
        
        # 질의응답 처리
        result = rag.query(
            question=request.question,
            top_k=request.top_k
        )
        
        # 정확한 KorQuAD 포맷 응답 처리
        return QAResponse(
            retrieved_document_id=result.get("retrieved_document_id", 0),
            retrieved_document=result.get("retrieved_document", ""),
            question=result.get("question", request.question),
            answers=result.get("answers", "답변을 생성할 수 없습니다."),
            quality_metrics=result.get("quality_metrics")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"질의응답 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/system/info")
async def get_system_info():
    """시스템 정보 조회 엔드포인트"""
    try:
        rag = get_rag_chain()
        return rag.get_system_info()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"시스템 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/loading/status")
async def get_loading_status():
    """데이터 로딩 상태 조회 엔드포인트"""
    return data_loader.get_status()

@router.post("/loading/initialize")
async def initialize_data(data_path: Optional[str] = None):
    """데이터 초기화 엔드포인트 (관리자용)"""
    if data_loader.is_loading:
        return {"message": "이미 로딩 중입니다.", "status": data_loader.get_status()}
    
    if data_loader.is_ready:
        return {"message": "이미 초기화 완료되었습니다.", "status": data_loader.get_status()}
    
    # 비동기 초기화 시작
    import asyncio
    asyncio.create_task(data_loader.initialize(data_path))
    
    return {
        "message": "데이터 초기화를 시작했습니다.",
        "status": data_loader.get_status()
    }
