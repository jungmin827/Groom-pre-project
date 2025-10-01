"""애플리케이션 설정 관리 모듈"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # ChromaDB 설정
    chroma_dir: str = "./chroma_db"
    
    # 임베딩 모델 설정
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # LLM 모델 설정
    llm_model: str = "microsoft/DialoGPT-medium"
    
    # 서버 설정
    port: int = 8000
    host: str = "0.0.0.0"
    
    # API 설정
    api_title: str = "KorQuAD RAG API"
    api_description: str = "한국어 질의응답을 위한 RAG 시스템 API"
    api_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 전역 설정 인스턴스
settings = Settings()
