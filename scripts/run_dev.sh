#!/bin/bash

# KorQuAD RAG 개발 환경 실행 스크립트

set -e  # 에러 발생 시 스크립트 종료

echo "🚀 KorQuAD RAG 개발 환경 설정을 시작합니다..."

# 가상환경 생성 (이미 존재하지 않는 경우)
if [ ! -d ".venv" ]; then
    echo "📦 가상환경을 생성합니다..."
    python -m venv .venv
else
    echo "✅ 가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "🔧 가상환경을 활성화합니다..."
source .venv/bin/activate

# 의존성 설치
echo "📥 의존성을 설치합니다..."
pip install --upgrade pip
pip install -r requirements.txt

# 테스트 실행
echo "🧪 테스트를 실행합니다..."
pytest -q

echo "✅ 테스트가 성공적으로 완료되었습니다!"

# 개발 서버 실행
echo "🌐 개발 서버를 시작합니다..."
echo "📖 API 문서: http://localhost:8000/docs"
echo "🔍 ReDoc: http://localhost:8000/redoc"
echo "💡 서버를 중지하려면 Ctrl+C를 누르세요"
echo ""

uvicorn src.app.main:app --reload --port 8000 --host 0.0.0.0
