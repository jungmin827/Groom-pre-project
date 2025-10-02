"""
벡터 인덱스 관리 모듈
검색 정확도와 효율성을 높이는 인덱싱 시스템
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
import os
import json
from pathlib import Path

from app.embeddings.embeddings import EmbeddingModel
from app.data.preprocess import load_and_process_korquad

class VectorIndex:
    """벡터 인덱스 관리 클래스"""
    
    def __init__(self, 
                 index_dir: str = "./vector_index",
                 collection_name: str = "korquad_docs",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 score_threshold: float = 0.6):
        """
        벡터 인덱스 초기화
        
        Args:
            index_dir: 인덱스 저장 디렉토리
            collection_name: ChromaDB 컬렉션 이름
            model_name: 임베딩 모델 이름
            score_threshold: 유사도 점수 임계값
        """
        self.index_dir = index_dir
        self.collection_name = collection_name
        self.score_threshold = score_threshold
        
        # 디렉토리 생성
        os.makedirs(index_dir, exist_ok=True)
        
        # 임베딩 모델 초기화
        self.embedding_model = EmbeddingModel(model_name)
        
        # ChromaDB 클라이언트 초기화
        self.client = None
        self.collection = None
        self._initialize_chroma()
        
    def _initialize_chroma(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.index_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 가져오기
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                print(f"✅ 기존 컬렉션 로드: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "KorQuAD 문서 컬렉션"}
                )
                print(f"✅ 새 컬렉션 생성: {self.collection_name}")
                
        except Exception as e:
            raise RuntimeError(f"ChromaDB 초기화 실패: {str(e)}")
    
    def build_index(self, data_path: str, chunk_size: int = 500, overlap: int = 50) -> Dict[str, Any]:
        """
        최초 1회 전처리한 KorQuAD 데이터를 인덱싱
        
        Args:
            data_path: KorQuAD 데이터 파일 경로
            chunk_size: 청크 크기
            overlap: 겹치는 부분 크기
            
        Returns:
            인덱싱 결과 정보
        """
        print("🚀 벡터 인덱스 구축 시작...")
        
        try:
            # 1. 데이터 로드 및 전처리
            print("📂 데이터 로드 및 전처리 중...")
            documents, metadatas, doc_ids = load_and_process_korquad(
                data_path, chunk_size, overlap
            )
            
            if not documents:
                raise ValueError("전처리된 데이터가 없습니다.")
            
            print(f"✅ {len(documents)}개 청크 생성 완료")
            
            # 2. 임베딩 생성
            print("🧠 임베딩 생성 중...")
            embeddings = self.embedding_model.encode(documents, batch_size=32)
            print(f"✅ {embeddings.shape[0]}개 임베딩 생성 완료")
            
            # 3. ChromaDB에 저장
            print("💾 벡터 인덱스 저장 중...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids,
                embeddings=embeddings.tolist()
            )
            
            # 4. 인덱스 메타데이터 저장
            index_metadata = {
                "total_documents": len(documents),
                "embedding_dimension": embeddings.shape[1],
                "model_name": self.embedding_model.model_name,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "score_threshold": self.score_threshold,
                "data_path": data_path
            }
            
            self._save_index_metadata(index_metadata)
            
            print("✅ 벡터 인덱스 구축 완료!")
            return index_metadata
            
        except Exception as e:
            raise RuntimeError(f"인덱스 구축 실패: {str(e)}")
    
    def persist(self) -> bool:
        """
        로컬 디렉토리에 저장 (재시작해도 유지)
        
        Returns:
            저장 성공 여부
        """
        try:
            # ChromaDB는 자동으로 영구 저장되므로 메타데이터만 저장
            metadata_path = os.path.join(self.index_dir, "index_metadata.json")
            
            if os.path.exists(metadata_path):
                print(f"✅ 인덱스가 이미 저장되어 있습니다: {self.index_dir}")
                return True
            else:
                print("⚠️ 저장할 인덱스 메타데이터가 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 인덱스 저장 실패: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """
        서버 재시작 시 기존 인덱스를 다시 로드
        
        Returns:
            로드 성공 여부
        """
        try:
            # ChromaDB 컬렉션 로드 확인
            if self.collection is None:
                return False
            
            # 컬렉션 정보 확인
            count = self.collection.count()
            if count == 0:
                print("⚠️ 빈 컬렉션입니다.")
                return False
            
            # 메타데이터 로드
            metadata = self._load_index_metadata()
            if metadata:
                print(f"✅ 인덱스 로드 완료: {count}개 문서")
                print(f"   - 모델: {metadata.get('model_name', 'N/A')}")
                print(f"   - 차원: {metadata.get('embedding_dimension', 'N/A')}")
                print(f"   - 임계값: {metadata.get('score_threshold', 'N/A')}")
                return True
            else:
                print("⚠️ 인덱스 메타데이터를 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 인덱스 로드 실패: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        질문 입력 시 top_k 문서 반환 (출처 정보와 함께)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            score_threshold: 유사도 점수 임계값 (None이면 기본값 사용)
            
        Returns:
            검색 결과 리스트
        """
        if not query:
            return []
        
        try:
            # 임계값 설정
            threshold = score_threshold if score_threshold is not None else self.score_threshold
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode_single(query)
            
            # ChromaDB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅 및 필터링
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 거리를 유사도 점수로 변환 (0-1 범위)
                    similarity_score = 1 - distance
                    
                    # 임계값 필터링
                    if similarity_score < threshold:
                        continue
                    
                    # 출처 정보 포함
                    formatted_results.append({
                        "id": metadata.get("id", f"result_{i}"),
                        "title": metadata.get("title", "제목 없음"),
                        "content": doc,
                        "score": round(similarity_score, 4),
                        "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                        "metadata": {
                            "original_id": metadata.get("original_id", ""),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "start": metadata.get("start", 0),
                            "end": metadata.get("end", 0),
                            "sentence_count": metadata.get("sentence_count", 0),
                            "source": metadata.get("source", "korquad")
                        }
                    })
            
            return formatted_results
            
        except Exception as e:
            raise RuntimeError(f"검색 실패: {str(e)}")
    
    def _save_index_metadata(self, metadata: Dict[str, Any]):
        """인덱스 메타데이터 저장"""
        try:
            metadata_path = os.path.join(self.index_dir, "index_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 메타데이터 저장 실패: {str(e)}")
    
    def _load_index_metadata(self) -> Optional[Dict[str, Any]]:
        """인덱스 메타데이터 로드"""
        try:
            metadata_path = os.path.join(self.index_dir, "index_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"⚠️ 메타데이터 로드 실패: {str(e)}")
            return None
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        인덱스 정보 조회
        
        Returns:
            인덱스 정보 딕셔너리
        """
        try:
            count = self.collection.count() if self.collection else 0
            metadata = self._load_index_metadata()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "index_dir": self.index_dir,
                "score_threshold": self.score_threshold,
                "embedding_model": self.embedding_model.get_model_info(),
                "metadata": metadata
            }
        except Exception as e:
            return {"error": f"인덱스 정보 조회 실패: {str(e)}"}
    
    def update_score_threshold(self, new_threshold: float):
        """
        유사도 점수 임계값 업데이트
        
        Args:
            new_threshold: 새로운 임계값
        """
        if 0.0 <= new_threshold <= 1.0:
            self.score_threshold = new_threshold
            print(f"✅ 임계값 업데이트: {new_threshold}")
        else:
            raise ValueError("임계값은 0.0과 1.0 사이여야 합니다.")
    
    def clear_index(self):
        """인덱스 초기화"""
        try:
            if self.collection:
                # 컬렉션 삭제
                self.client.delete_collection(self.collection_name)
                
                # 새 컬렉션 생성
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "KorQuAD 문서 컬렉션"}
                )
                
                # 메타데이터 파일 삭제
                metadata_path = os.path.join(self.index_dir, "index_metadata.json")
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                print("✅ 인덱스 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"인덱스 초기화 실패: {str(e)}")
