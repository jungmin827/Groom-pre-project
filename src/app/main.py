"""FastAPI 애플리케이션 메인 모듈"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 생성"""
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS 미들웨어 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # API 라우터 등록
    app.include_router(router, prefix="/api/v1")
    
    return app


# 애플리케이션 인스턴스 생성
app = create_app()


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "KorQuAD RAG API",
        "version": settings.api_version,
        "docs": "/docs"
    }
