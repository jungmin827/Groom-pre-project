"""
비동기 데이터 로더 모듈
서버 시작 시 데이터셋 로딩과 임베딩, DB 구축을 비동기로 처리
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from app.data.preprocess import load_and_process_korquad, validate_chunk_quality
from app.retriever.retriever import DocumentRetriever
from app.embeddings.embeddings import EmbeddingModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class AsyncDataLoader:
    """비동기 데이터 로더 클래스"""
    
    def __init__(self):
        self.is_loading = False
        self.is_ready = False
        self.loading_progress = 0.0
        self.loading_status = "대기 중"
        self.error_message = None
        self.start_time = None
        self.retriever = None
        self.embedding_model = None
        
    async def initialize(self, data_path: Optional[str] = None) -> bool:
        """
        비동기 데이터 초기화
        
        Args:
            data_path: 데이터 파일 경로 (None이면 기존 DB 사용)
            
        Returns:
            초기화 성공 여부
        """
        if self.is_loading:
            logger.warning("이미 로딩 중입니다.")
            return False
            
        if self.is_ready:
            logger.info("이미 초기화 완료되었습니다.")
            return True
            
        self.is_loading = True
        self.start_time = datetime.now()
        self.loading_progress = 0.0
        self.loading_status = "초기화 시작"
        self.error_message = None
        
        try:
            # 1. Retriever 초기화
            await self._update_status("Retriever 초기화 중...", 0.1)
            self.retriever = DocumentRetriever(settings.chroma_dir)
            
            # 2. 기존 데이터 확인
            collection_info = self.retriever.get_collection_info()
            if collection_info.get("document_count", 0) > 0:
                await self._update_status("기존 데이터베이스 로드 중...", 0.3)
                logger.info(f"기존 데이터베이스 발견: {collection_info['document_count']}개 문서")
                self.is_ready = True
                self.is_loading = False
                return True
            
            # 3. 새 데이터 로딩 (data_path가 제공된 경우)
            if data_path and Path(data_path).exists():
                await self._load_new_data(data_path)
            else:
                await self._update_status("데이터 파일을 찾을 수 없습니다.", 1.0)
                self.error_message = "데이터 파일을 찾을 수 없습니다."
                return False
                
            self.is_ready = True
            self.is_loading = False
            await self._update_status("초기화 완료", 1.0)
            return True
            
        except Exception as e:
            logger.error(f"데이터 초기화 실패: {str(e)}")
            self.error_message = str(e)
            self.is_loading = False
            return False
    
    async def _load_new_data(self, data_path: str):
        """새 데이터 로딩 및 처리"""
        try:
            # 1. 데이터 전처리
            await self._update_status("데이터 전처리 중...", 0.2)
            documents, metadatas, doc_ids = load_and_process_korquad(
                data_path, 
                chunk_size=500, 
                overlap=50
            )
            
            if not documents:
                raise ValueError("전처리된 데이터가 없습니다.")
            
            # 2. 청크 품질 검증
            await self._update_status("청크 품질 검증 중...", 0.4)
            quality_result = validate_chunk_quality(documents)
            logger.info(f"청크 품질 점수: {quality_result.get('quality_score', 0):.1f}%")
            
            # 3. 임베딩 모델 초기화
            await self._update_status("임베딩 모델 로딩 중...", 0.5)
            self.embedding_model = EmbeddingModel(settings.embedding_model)
            
            # 4. 배치 임베딩 생성
            await self._update_status("임베딩 생성 중...", 0.7)
            batch_size = 32
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = doc_ids[i:i + batch_size]
                
                # 임베딩 생성
                embeddings = self.embedding_model.encode(batch_docs, batch_size=batch_size)
                
                # ChromaDB에 추가
                self.retriever.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=embeddings.tolist()
                )
                
                # 진행률 업데이트
                progress = 0.7 + (i / len(documents)) * 0.2
                await self._update_status(f"임베딩 생성 중... ({i+len(batch_docs)}/{len(documents)})", progress)
            
            # 5. 최종 검증
            await self._update_status("최종 검증 중...", 0.95)
            final_info = self.retriever.get_collection_info()
            logger.info(f"데이터 로딩 완료: {final_info.get('document_count', 0)}개 문서")
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            raise
    
    async def _update_status(self, status: str, progress: float):
        """로딩 상태 업데이트"""
        self.loading_status = status
        self.loading_progress = progress
        logger.info(f"[{progress*100:.1f}%] {status}")
        await asyncio.sleep(0.01)  # 다른 태스크에게 제어권 양보
    
    def get_status(self) -> Dict[str, Any]:
        """현재 로딩 상태 반환"""
        elapsed_time = None
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "is_loading": self.is_loading,
            "is_ready": self.is_ready,
            "loading_progress": self.loading_progress,
            "loading_status": self.loading_status,
            "error_message": self.error_message,
            "elapsed_time": elapsed_time,
            "retriever_info": self.retriever.get_collection_info() if self.retriever else None
        }
    
    def get_retriever(self) -> Optional[DocumentRetriever]:
        """Retriever 인스턴스 반환"""
        return self.retriever if self.is_ready else None

# 전역 데이터 로더 인스턴스
data_loader = AsyncDataLoader()
