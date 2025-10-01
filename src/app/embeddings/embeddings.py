"""임베딩 생성 및 관리 모듈"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class EmbeddingService:
    """임베딩 서비스 클래스"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        임베딩 서비스 초기화
        
        Args:
            model_name (Optional[str]): 사용할 임베딩 모델명
        """
        self.model_name = model_name or settings.embedding_model
        self.model = None
        
    def load_model(self) -> None:
        """임베딩 모델을 로드합니다."""
        # TODO: 실제 모델 로딩 구현
        # self.model = SentenceTransformer(self.model_name)
        pass
        
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        텍스트 또는 텍스트 리스트에 대한 임베딩을 생성합니다.
        
        Args:
            texts (Union[str, List[str]]): 임베딩을 생성할 텍스트(들)
            
        Returns:
            np.ndarray: 생성된 임베딩 벡터(들)
        """
        if not self.model:
            self.load_model()
            
        # TODO: 실제 임베딩 생성 구현
        # if isinstance(texts, str):
        #     texts = [texts]
        # embeddings = self.model.encode(texts)
        # return embeddings
        
        # 스텁 구현: 랜덤 임베딩 반환
        if isinstance(texts, str):
            return np.random.rand(384)  # 일반적인 임베딩 차원
        else:
            return np.random.rand(len(texts), 384)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 벡터 간의 코사인 유사도를 계산합니다.
        
        Args:
            embedding1 (np.ndarray): 첫 번째 임베딩 벡터
            embedding2 (np.ndarray): 두 번째 임베딩 벡터
            
        Returns:
            float: 코사인 유사도 (0-1 범위)
        """
        # 코사인 유사도 계산
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        대량의 텍스트를 배치로 나누어 임베딩을 생성합니다.
        
        Args:
            texts (List[str]): 임베딩을 생성할 텍스트 리스트
            batch_size (int): 배치 크기
            
        Returns:
            np.ndarray: 생성된 임베딩 벡터들
        """
        if not self.model:
            self.load_model()
            
        # TODO: 실제 배치 임베딩 구현
        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch = texts[i:i + batch_size]
        #     batch_embeddings = self.model.encode(batch)
        #     embeddings.append(batch_embeddings)
        # return np.vstack(embeddings)
        
        # 스텁 구현
        return np.random.rand(len(texts), 384)


# 전역 임베딩 서비스 인스턴스
embedding_service = EmbeddingService()
