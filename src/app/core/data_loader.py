"""
비동기 데이터 로더 모듈
서버 시작 시 데이터셋 로딩과 임베딩, DB 구축을 비동기로 처리
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm

from app.data.preprocess import load_and_process_korquad
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
        """비동기 데이터 초기화"""
        if self.is_loading or self.is_ready:
            return True

        self.is_loading = True
        self.start_time = datetime.now()
        current_data_path = data_path or settings.data_path

        try:
            await self._update_status("Retriever 초기화 중...", 0.05)
            self.retriever = DocumentRetriever(settings.chroma_dir)

            collection_info = self.retriever.get_collection_info()
            if collection_info and collection_info.get("document_count", 0) > 0:
                await self._update_status("기존 데이터베이스 로드 완료.", 1.0)
                logger.info(f"기존 데이터베이스 발견: {collection_info.get('document_count', 0)}개 문서")
                self.is_ready = True
            else:
                await self._load_new_data(current_data_path)
                self.is_ready = True

            self.is_loading = False
            await self._update_status("초기화 완료", 1.0)
            return True
        except Exception as e:
            logger.error(f"데이터 초기화 실패: {str(e)}", exc_info=True)
            self.error_message = str(e)
            self.is_loading = False
            return False

    async def _load_new_data(self, data_path: str):
        """새 데이터 로딩 및 처리 (자동 다운로드 기능 포함)"""
        data_file = Path(data_path)
        if not data_file.exists():
            await self._download_data(data_file)

        await self._update_status("데이터 전처리 시작...", 0.1)
        documents, metadatas, doc_ids = load_and_process_korquad(
            data_path,
            chunk_size=settings.chunk_size,
            overlap=settings.overlap
        )
        if not documents:
            raise ValueError("전처리된 데이터가 없습니다.")

        await self._update_status("임베딩 모델 로딩 중...", 0.6)
        self.embedding_model = EmbeddingModel(settings.embedding_model)

        await self._update_status("임베딩 생성 및 DB 저장 시작...", 0.7)
        await self._embed_and_store_data(documents, metadatas, doc_ids)

        await self._update_status("최종 검증 중...", 0.95)
        final_info = self.retriever.get_collection_info()
        logger.info(f"데이터 로딩 완료: {final_info.get('document_count', 0)}개 문서")

    async def _download_data(self, data_file: Path):
        """데이터셋 파일 다운로드"""
        logger.info(f"데이터 파일을 찾을 수 없습니다: {data_file}. 다운로드를 시작합니다.")
        data_url = "https://korquad.github.io/dataset/KorQuAD_v1.0_train.json"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(data_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(data_file, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="KorQuAD 다운로드 중"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            logger.info("데이터 파일 다운로드 완료.")
        except Exception as e:
            logger.error(f"데이터 파일 다운로드 실패: {e}")
            if data_file.exists(): data_file.unlink()
            raise

    async def _embed_and_store_data(self, documents, metadatas, doc_ids):
        """데이터를 배치로 임베딩하고 DB에 저장"""
        batch_size = 32
        for i in tqdm(range(0, len(documents), batch_size), desc="임베딩 및 DB 저장"):
            batch_docs = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]

            embeddings = self.embedding_model.encode(batch_docs, show_progress=False)

            self.retriever.collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=embeddings.tolist()
            )
            progress = 0.7 + ((i + len(batch_docs)) / len(documents)) * 0.25
            await self._update_status(f"DB 저장 중... ({i+len(batch_docs)}/{len(documents)})", progress)

    async def _update_status(self, status: str, progress: float):
        self.loading_status = status
        self.loading_progress = round(progress, 2)
        logger.info(f"[{self.loading_progress * 100:.1f}%] {status}")
        await asyncio.sleep(0.01)

    def get_status(self) -> Dict[str, Any]:
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            "is_loading": self.is_loading,
            "is_ready": self.is_ready,
            "loading_progress": self.loading_progress,
            "loading_status": self.loading_status,
            "error_message": self.error_message,
            "elapsed_time": round(elapsed_time, 2),
            "retriever_info": self.retriever.get_collection_info() if self.retriever else None
        }

    def get_retriever(self) -> Optional[DocumentRetriever]:
        return self.retriever if self.is_ready else None

data_loader = AsyncDataLoader()
