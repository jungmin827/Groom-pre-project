"""
문서 검색 모듈
"""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
import os
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class DocumentRetriever:
    """문서 검색 클래스"""
    
    def __init__(self, chroma_dir: str = None):
        """
        문서 검색기 초기화
        
        Args:
            chroma_dir: ChromaDB 저장 디렉토리
        """
        if chroma_dir is None:
            # 기본값으로 프로젝트 루트의 chroma_db 디렉토리 사용
            chroma_dir = str(PROJECT_ROOT / "chroma_db")

        self.chroma_dir = chroma_dir
        self.client = None
        self.collection = None

        self._initialize_chroma()

    def _initialize_chroma(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            # ChromaDB 디렉토리 생성
            os.makedirs(self.chroma_dir, exist_ok=True)
            
            # ChromaDB 클라이언트 설정
            self.client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 기본 컬렉션 생성 또는 가져오기
            self.create_collection()
            
        except Exception as e:
            raise RuntimeError(f"ChromaDB 클라이언트 초기화 실패: {str(e)}")
        
    def create_collection(self, collection_name: str = "wikipedia_docs"):
        """
        컬렉션 생성
        
        Args:
            collection_name: 컬렉션 이름
        """
        try:
            # 기존 컬렉션이 있으면 가져오기, 없으면 생성
            try:
                self.collection = self.client.get_collection(name=collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "위키피디아 문서 컬렉션"}
                )
        except Exception as e:
            raise RuntimeError(f"컬렉션 생성/가져오기 실패: {str(e)}")
        
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        문서를 벡터 데이터베이스에 추가
        
        Args:
            documents: 추가할 문서 리스트
            metadatas: 문서 메타데이터 리스트
        """
        if not self.collection:
            raise RuntimeError("컬렉션이 초기화되지 않았습니다.")
            
        try:
            # 메타데이터가 없으면 기본값 생성
            if metadatas is None:
                metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
            
            # 문서 ID 생성
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 문서 추가
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            
        except Exception as e:
            raise RuntimeError(f"문서 추가 실패: {str(e)}")
    
    def add_korquad_data(self, korquad_items: List[Dict[str, Any]]):
        """
        KorQuAD 데이터셋을 벡터 데이터베이스에 추가 (기존 방식)
        
        Args:
            korquad_items: KorQuAD 데이터 아이템 리스트
        """
        if not self.collection:
            raise RuntimeError("컬렉션이 초기화되지 않았습니다.")
            
        try:
            documents = []
            metadatas = []
            doc_ids = []
            
            for i, item in enumerate(korquad_items):
                # KorQuAD 아이템에서 context를 문서로 사용
                context = item.get("context", "")
                title = item.get("title", "")
                item_id = item.get("id", f"item_{i}")
                
                documents.append(context)
                metadatas.append({
                    "id": item_id,
                    "title": title,
                    "source": "korquad"
                })
                doc_ids.append(item_id)
            
            # 문서 추가
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            
        except Exception as e:
            raise RuntimeError(f"KorQuAD 데이터 추가 실패: {str(e)}")
    
    def add_processed_chunks(self, documents: List[str], metadatas: List[Dict[str, Any]], doc_ids: List[str]):
        """
        전처리된 청크를 벡터 데이터베이스에 추가
        
        Args:
            documents: 문서 내용 리스트
            metadatas: 메타데이터 리스트
            doc_ids: 문서 ID 리스트
        """
        if not self.collection:
            raise RuntimeError("컬렉션이 초기화되지 않았습니다.")
            
        try:
            # 문서 추가
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            
        except Exception as e:
            raise RuntimeError(f"전처리된 청크 추가 실패: {str(e)}")
        
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        유사 문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self.collection:
            raise RuntimeError("컬렉션이 초기화되지 않았습니다.")
            
        try:
            # 유사 문서 검색
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 거리를 유사도 점수로 변환 (0-1 범위)
                    similarity_score = 1 - distance
                    
                    # 위키피디아 문서 메타데이터 추출
                    title = metadata.get("title", "제목 없음")
                    url = metadata.get("url", "")
                    category = metadata.get("category", "")
                    
                    # 원본 KorQuAD ID 추출 (original_id 또는 id 필드에서)
                    original_id = metadata.get("original_id", metadata.get("id", f"doc_{i}"))
                    
                    formatted_results.append({
                        "retrieved_document_id": original_id,  # 원본 KorQuAD ID 사용
                        "retrieved_document": doc,  # KorQuAD 포맷에 맞게 수정
                        "metadata": metadata,
                        "title": title,
                        "url": url,
                        "category": category,
                        "score": round(similarity_score, 4),
                        "snippet": doc[:200] + "..." if len(doc) > 200 else doc
                    })
            
            return formatted_results
            
        except Exception as e:
            raise RuntimeError(f"문서 검색 실패: {str(e)}")
        
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        if not self.collection:
            return {"error": "컬렉션이 초기화되지 않았습니다."}
            
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection.name,
                "document_count": count,
                "chroma_dir": self.chroma_dir
            }
        except Exception as e:
            return {"error": f"컬렉션 정보 조회 실패: {str(e)}"}
