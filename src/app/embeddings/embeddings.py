"""
임베딩 생성 모듈
검색 정확도와 효율성을 높이는 임베딩 시스템
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

class EmbeddingModel:
    """임베딩 모델 래퍼 클래스"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        임베딩 모델 초기화
        
        Args:
            model_name: 사용할 임베딩 모델 이름
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def load_model(self):
        """모델 로딩"""
        try:
            print(f"🤖 임베딩 모델 로딩 중: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            print(f"✅ 임베딩 모델 로딩 완료 (디바이스: {self.device})")
        except Exception as e:
            raise RuntimeError(f"임베딩 모델 로딩 실패: {str(e)}")
        
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부
            
        Returns:
            임베딩 벡터 배열
        """
        if not texts:
            return np.array([])
        
        try:
            # 배치 처리로 메모리 효율성 향상
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                device=self.device
            )
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"임베딩 생성 실패: {str(e)}")
        
    def encode_single(self, text: str) -> np.ndarray:
        """
        단일 텍스트를 임베딩으로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        if not text:
            return np.array([])
        
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                device=self.device
            )
            
            return embedding[0]  # 단일 벡터 반환
            
        except Exception as e:
            raise RuntimeError(f"단일 텍스트 임베딩 생성 실패: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        임베딩 차원 수 반환
        
        Returns:
            임베딩 차원 수
        """
        if self.model is None:
            return 0
        
        # 테스트 텍스트로 차원 확인
        test_embedding = self.encode_single("test")
        return len(test_embedding)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.model is not None,
            "embedding_dimension": self.get_embedding_dimension()
        }
