"""
RAG Chain 통합 모듈
"""
from typing import List, Dict, Any, Optional
from app.retriever.retriever import DocumentRetriever
from app.models.llm import LLMWrapper
from app.core.config import settings
from app.core.data_loader import data_loader
from app.core.search_quality import SearchQualityManager

class RAGChain:
    """RAG Chain 통합 클래스"""
    
    def __init__(self, chroma_dir: str = None, llm_model: str = None):
        """
        RAG Chain 초기화 (비동기 로더 사용)
        
        Args:
            chroma_dir: ChromaDB 디렉토리
            llm_model: LLM 모델명
        """
        self.chroma_dir = chroma_dir or settings.chroma_dir
        self.llm_model = llm_model or settings.llm_model
        
        # 컴포넌트 초기화 (비동기 로더에서 retriever 가져오기)
        self.retriever = data_loader.get_retriever()
        self.llm = LLMWrapper(self.llm_model)
        self.quality_manager = SearchQualityManager()
        
        # 데이터 로더가 준비되지 않은 경우 경고
        if not self.retriever:
            raise RuntimeError("데이터 로더가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")
        
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        RAG 질의응답 처리 (정확한 KorQuAD 포맷)
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 수
            
        Returns:
            정확한 KorQuAD 포맷 응답
        """
        try:
            # 1. 문서 검색
            raw_search_results = self.retriever.search(question, n_results=top_k * 2)  # 더 많이 검색
            
            if not raw_search_results:
                return {
                    "retrieved_document_id": 0,
                    "retrieved_document": "",
                    "question": question,
                    "answers": "관련 문서를 찾을 수 없습니다."
                }
            
            # 2. 검색 결과 품질 필터링
            filtered_results = self.quality_manager.filter_search_results(raw_search_results, question)
            
            if not filtered_results:
                return {
                    "retrieved_document_id": 0,
                    "retrieved_document": "",
                    "question": question,
                    "answers": "관련성이 높은 문서를 찾을 수 없습니다."
                }
            
            # 3. 상위 결과만 사용 (품질이 검증된 결과)
            search_results = filtered_results[:top_k]
            
            # 4. 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 5. LLM을 통한 답변 생성
            rag_result = self.llm.generate_answer(
                question=question,
                context=context,
                sources=search_results
            )
            
            # 6. 답변 품질 검증
            quality_check = self.quality_manager.validate_answer_quality(
                question, rag_result.get("answer", ""), context
            )
            
            # 7. 정확한 KorQuAD 포맷으로 변환
            response = self._format_korquad_response(question, rag_result, search_results)
            
            # 8. 품질 정보 추가
            response["quality_metrics"] = {
                "confidence": quality_check.get("confidence", 0.0),
                "is_valid": quality_check.get("is_valid", False),
                "search_quality": self.quality_manager.get_quality_metrics(search_results)
            }
            
            return response
            
        except Exception as e:
            return {
                "retrieved_document_id": 0,
                "retrieved_document": "",
                "question": question,
                "answers": f"처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        검색 결과로부터 컨텍스트 구성 (위키피디아 문서 특화)
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            구성된 컨텍스트 문자열
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            score = result.get("score", 0.0)
            title = result.get("title", f"문서 {i}")
            category = result.get("category", "")
            
            # 위키피디아 문서 정보 포함
            doc_info = f"제목: {title}"
            if category:
                doc_info += f" | 분류: {category}"
            doc_info += f" | 유사도: {score:.3f}"
            
            context_parts.append(f"[위키피디아 문서 {i}] {doc_info}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_korquad_response(self, question: str, rag_result: Dict[str, Any], search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        정확한 KorQuAD 포맷에 맞는 응답 생성
        
        Args:
            question: 원본 질문
            rag_result: RAG 처리 결과
            search_results: 검색된 문서들
            
        Returns:
            정확한 KorQuAD 포맷 응답
        """
        # 첫 번째 검색 결과를 기준으로 정확한 포맷 생성
        if search_results:
            first_result = search_results[0]
            # 답변 검증 및 정제
            validated_answer = self._validate_and_refine_answer(
                rag_result.get("answer", ""), 
                question, 
                first_result.get("retrieved_document", "")
            )
            
            return {
                "retrieved_document_id": first_result.get("retrieved_document_id", 1),
                "retrieved_document": first_result.get("retrieved_document", ""),
                "question": question,
                "answers": validated_answer
            }
        else:
            return {
                "retrieved_document_id": 0,
                "retrieved_document": "",
                "question": question,
                "answers": "관련 문서를 찾을 수 없습니다."
            }
    
    def _validate_and_refine_answer(self, generated_answer: str, question: str, context: str) -> str:
        """
        생성된 답변을 검증하고 정제
        
        Args:
            generated_answer: LLM이 생성한 답변
            question: 원본 질문
            context: 검색된 컨텍스트
            
        Returns:
            검증되고 정제된 답변
        """
        if not generated_answer or generated_answer.strip() == "":
            return "답변을 생성할 수 없습니다."
        
        # 답변 정제
        answer = generated_answer.strip()
        
        # 답변이 너무 길면 요약
        if len(answer) > 200:
            # 문장 단위로 분할하여 첫 번째 문장만 사용
            sentences = answer.split('.')
            if sentences:
                answer = sentences[0].strip() + "."
        
        # 답변이 컨텍스트에 기반하는지 확인
        if not self._is_answer_relevant(answer, context):
            return "제공된 문서에서 해당 정보를 찾을 수 없습니다."
        
        return answer
    
    def _is_answer_relevant(self, answer: str, context: str) -> bool:
        """
        답변이 컨텍스트와 관련이 있는지 확인 (개선된 로직)
        
        Args:
            answer: 생성된 답변
            context: 검색된 컨텍스트
            
        Returns:
            관련성 여부
        """
        if not answer or not context:
            return False
        
        # 품질 관리자를 통한 정교한 관련성 검사
        quality_check = self.quality_manager.validate_answer_quality("", answer, context)
        return quality_check.get("is_valid", False)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        시스템 정보 조회
        
        Returns:
            시스템 정보 딕셔너리
        """
        return {
            "retriever": self.retriever.get_collection_info(),
            "llm": self.llm.get_model_info(),
            "chroma_dir": self.chroma_dir,
            "llm_model": self.llm_model
        }
