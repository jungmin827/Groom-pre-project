"""
검색 품질 관리 및 관련도 평가 모듈
"""
import re
from typing import List, Dict, Any
from collections import Counter
import numpy as np

class SearchQualityManager:
    """검색 품질 관리 클래스"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 relevance_threshold: float = 0.3,
                 min_keyword_overlap: int = 2):
        """
        검색 품질 관리자 초기화
        
        Args:
            similarity_threshold: 유사도 임계값
            relevance_threshold: 관련도 임계값
            min_keyword_overlap: 최소 키워드 겹침 수
        """
        self.similarity_threshold = similarity_threshold
        self.relevance_threshold = relevance_threshold
        self.min_keyword_overlap = min_keyword_overlap
    
    def filter_search_results(self, 
                             search_results: List[Dict[str, Any]], 
                             query: str) -> List[Dict[str, Any]]:
        """
        검색 결과를 품질 기준으로 필터링
        
        Args:
            search_results: 원본 검색 결과
            query: 검색 쿼리
            
        Returns:
            필터링된 검색 결과
        """
        if not search_results:
            return []
        
        filtered_results = []
        query_keywords = self._extract_keywords(query)
        
        for result in search_results:
            # 1. 유사도 점수 검증
            if result.get('score', 0) < self.similarity_threshold:
                continue
            
            # 2. 키워드 관련성 검증
            doc_keywords = self._extract_keywords(result.get('retrieved_document', ''))
            keyword_overlap = len(query_keywords.intersection(doc_keywords))
            
            if keyword_overlap < self.min_keyword_overlap:
                continue
            
            # 3. 의미적 관련성 검증
            relevance_score = self._calculate_relevance_score(query, result)
            if relevance_score < self.relevance_threshold:
                continue
            
            # 4. 품질 점수 추가
            result['relevance_score'] = relevance_score
            result['keyword_overlap'] = keyword_overlap
            filtered_results.append(result)
        
        # 품질 점수 기준으로 정렬
        filtered_results.sort(key=lambda x: x.get('score', 0) * 0.7 + x.get('relevance_score', 0) * 0.3, reverse=True)
        
        return filtered_results
    
    def _extract_keywords(self, text: str) -> set:
        """
        텍스트에서 키워드 추출 (한국어 특성 고려)
        
        Args:
            text: 추출할 텍스트
            
        Returns:
            키워드 집합
        """
        if not text:
            return set()
        
        # 한국어 불용어 제거
        stopwords = {
            '이', '가', '을', '를', '에', '의', '와', '과', '은', '는', '도', '로', '으로',
            '에서', '에게', '한테', '부터', '까지', '처럼', '같이', '만', '조차', '마저',
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로',
            '에서', '에게', '한테', '부터', '까지', '처럼', '같이', '만', '조차', '마저',
            '그', '이', '저', '그것', '이것', '저것', '그런', '이런', '저런',
            '하다', '되다', '있다', '없다', '같다', '이다', '아니다'
        }
        
        # 텍스트 정리
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        words = text.split()
        
        # 불용어 제거 및 길이 필터링
        keywords = {word for word in words 
                   if len(word) > 1 and word not in stopwords}
        
        return keywords
    
    def _calculate_relevance_score(self, query: str, result: Dict[str, Any]) -> float:
        """
        의미적 관련성 점수 계산
        
        Args:
            query: 검색 쿼리
            result: 검색 결과
            
        Returns:
            관련성 점수 (0-1)
        """
        doc_text = result.get('retrieved_document', '')
        if not doc_text:
            return 0.0
        
        # 1. 키워드 겹침 비율
        query_keywords = self._extract_keywords(query)
        doc_keywords = self._extract_keywords(doc_text)
        
        if not query_keywords:
            return 0.0
        
        keyword_overlap_ratio = len(query_keywords.intersection(doc_keywords)) / len(query_keywords)
        
        # 2. 문맥적 유사성 (간단한 TF-IDF 기반)
        context_similarity = self._calculate_context_similarity(query, doc_text)
        
        # 3. 제목 관련성
        title_relevance = self._calculate_title_relevance(query, result.get('title', ''))
        
        # 가중 평균으로 최종 점수 계산
        relevance_score = (
            keyword_overlap_ratio * 0.4 +
            context_similarity * 0.4 +
            title_relevance * 0.2
        )
        
        return min(relevance_score, 1.0)
    
    def _calculate_context_similarity(self, query: str, doc_text: str) -> float:
        """
        문맥적 유사성 계산
        
        Args:
            query: 검색 쿼리
            doc_text: 문서 텍스트
            
        Returns:
            문맥적 유사성 점수 (0-1)
        """
        query_words = self._extract_keywords(query)
        doc_words = self._extract_keywords(doc_text)
        
        if not query_words or not doc_words:
            return 0.0
        
        # 공통 단어의 빈도 고려
        query_counter = Counter(query_words)
        doc_counter = Counter(doc_words)
        
        common_words = query_words.intersection(doc_words)
        if not common_words:
            return 0.0
        
        # TF-IDF 유사한 방식으로 점수 계산
        similarity_score = 0.0
        for word in common_words:
            query_freq = query_counter[word]
            doc_freq = doc_counter[word]
            similarity_score += min(query_freq, doc_freq) / max(query_freq, doc_freq)
        
        return similarity_score / len(common_words)
    
    def _calculate_title_relevance(self, query: str, title: str) -> float:
        """
        제목 관련성 계산
        
        Args:
            query: 검색 쿼리
            title: 문서 제목
            
        Returns:
            제목 관련성 점수 (0-1)
        """
        if not title:
            return 0.0
        
        query_keywords = self._extract_keywords(query)
        title_keywords = self._extract_keywords(title)
        
        if not query_keywords or not title_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(title_keywords))
        return min(overlap / len(query_keywords), 1.0)
    
    def validate_answer_quality(self, 
                               question: str, 
                               answer: str, 
                               context: str) -> Dict[str, Any]:
        """
        답변 품질 검증
        
        Args:
            question: 질문
            answer: 답변
            context: 컨텍스트
            
        Returns:
            검증 결과
        """
        if not answer or not context:
            return {
                'is_valid': False,
                'reason': '답변 또는 컨텍스트가 없습니다.',
                'confidence': 0.0
            }
        
        # 1. 답변 길이 검증
        if len(answer.strip()) < 3:
            return {
                'is_valid': False,
                'reason': '답변이 너무 짧습니다.',
                'confidence': 0.0
            }
        
        # 2. 컨텍스트 관련성 검증
        context_relevance = self._calculate_relevance_score(question, {'retrieved_document': context})
        
        # 3. 답변-컨텍스트 일치성 검증
        answer_context_match = self._calculate_answer_context_match(answer, context)
        
        # 4. 질문-답변 관련성 검증
        qa_relevance = self._calculate_qa_relevance(question, answer)
        
        # 종합 점수 계산
        confidence = (context_relevance * 0.4 + 
                     answer_context_match * 0.3 + 
                     qa_relevance * 0.3)
        
        is_valid = confidence >= self.relevance_threshold
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'context_relevance': context_relevance,
            'answer_context_match': answer_context_match,
            'qa_relevance': qa_relevance,
            'reason': '답변이 컨텍스트와 관련이 없습니다.' if not is_valid else '답변이 유효합니다.'
        }
    
    def _calculate_answer_context_match(self, answer: str, context: str) -> float:
        """답변과 컨텍스트의 일치성 계산"""
        answer_keywords = self._extract_keywords(answer)
        context_keywords = self._extract_keywords(context)
        
        if not answer_keywords:
            return 0.0
        
        overlap = len(answer_keywords.intersection(context_keywords))
        return min(overlap / len(answer_keywords), 1.0)
    
    def _calculate_qa_relevance(self, question: str, answer: str) -> float:
        """질문과 답변의 관련성 계산"""
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        if not question_keywords or not answer_keywords:
            return 0.0
        
        overlap = len(question_keywords.intersection(answer_keywords))
        return min(overlap / len(question_keywords), 1.0)
    
    def get_quality_metrics(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        검색 결과의 품질 메트릭 계산
        
        Args:
            search_results: 검색 결과
            
        Returns:
            품질 메트릭
        """
        if not search_results:
            return {'error': '검색 결과가 없습니다.'}
        
        scores = [result.get('score', 0) for result in search_results]
        relevance_scores = [result.get('relevance_score', 0) for result in search_results]
        
        return {
            'total_results': len(search_results),
            'avg_similarity_score': np.mean(scores) if scores else 0,
            'avg_relevance_score': np.mean(relevance_scores) if relevance_scores else 0,
            'min_similarity_score': min(scores) if scores else 0,
            'max_similarity_score': max(scores) if scores else 0,
            'high_quality_results': len([s for s in scores if s >= self.similarity_threshold]),
            'quality_ratio': len([s for s in scores if s >= self.similarity_threshold]) / len(scores) if scores else 0
        }
