"""
FastAPI 애플리케이션 진입점
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from pathlib import Path

from app.api.routes import router
from app.core.data_loader import data_loader
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="KorQuAD RAG API",
    description="한국어 질의응답을 위한 RAG 시스템",
    version="0.1.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 마운트
app.include_router(router, prefix="/api/v1")

# 루트 경로
@app.get("/")
async def root():
    return {"message": "KorQuAD RAG API", "version": "0.1.0"}

# 서버 시작 시 비동기 데이터 초기화
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 비동기 데이터 초기화"""
    logger.info("서버 시작 - 데이터 초기화를 시작합니다.")
    
    # 비동기로 데이터 초기화 시작
    asyncio.create_task(data_loader.initialize())
    
    logger.info("데이터 초기화가 백그라운드에서 시작되었습니다.")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리 작업"""
    logger.info("서버 종료 중...")
