"""
애플리케이션 설정 관리
"""
from pydantic_settings import BaseSettings
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 데이터베이스 설정
    chroma_dir: str = str(PROJECT_ROOT / "chroma_db")

    # 데이터 설정
    data_path: str = str(PROJECT_ROOT / "data" / "korquad_v1.0_train.json")
    chunk_size: int = 500
    overlap: int = 50
    
    # 모델 설정
    embedding_model: str = "intfloat/multilingual-e5-small"
    llm_model: str = "Qwen/Qwen2-1.5B-Instruct"
    
    # 서버 설정
    port: int = 8000
    host: str = "0.0.0.0"
    
    # 기타 설정
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# 전역 설정 인스턴스
settings = Settings()
