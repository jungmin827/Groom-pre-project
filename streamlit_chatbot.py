import streamlit as st
import requests

st.set_page_config(page_title="RAG 챗봇", page_icon="🤖", layout="centered")

# 다크모드 스타일 커스텀 CSS
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #18181b !important;
        color: #e5e7eb !important;
    }
    .stTextInput > div > div > input {
        background: #27272a !important;
        color: #e5e7eb !important;
        border-radius: 8px;
        border: 1px solid #3f3f46;
    }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%) !important;
        color: #fff !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 0.5em 1.5em;
        margin-top: 0.5em;
    }
    .stMarkdown, .stExpander, .stInfo {
        background: #23232a !important;
        color: #e5e7eb !important;
        border-radius: 8px;
        padding: 0.7em 1em;
        margin-bottom: 0.5em;
    }
    .user-msg {
        background: #27272a;
        color: #facc15;
        border-radius: 8px;
        padding: 0.7em 1em;
        margin-bottom: 0.3em;
        border-left: 4px solid #facc15;
    }
    .bot-msg {
        background: #23232a;
        color: #38bdf8;
        border-radius: 8px;
        padding: 0.7em 1em;
        margin-bottom: 0.3em;
        border-left: 4px solid #38bdf8;
    }
    .stExpander > div {
        background: #18181b !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 korquad데이터셋 기반 RAG 챗봇 ")

# 세션 상태에 대화 기록 저장
def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return st.session_state["chat_history"]

chat_history = get_chat_history()

# 질문 입력창
user_input = st.text_input("질문을 입력하세요:", key="input_box")

# 로딩 상태 관리
def set_loading(val: bool):
    st.session_state["loading"] = val
if "loading" not in st.session_state:
    st.session_state["loading"] = False

# 전송 버튼
if st.button("전송") and user_input.strip():
    API_URL = "http://localhost:8000/qa"  # 실제 환경에 맞게 수정
    payload = {"question": user_input, "top_k": 3}
    set_loading(True)
    try:
        with st.spinner("답변을 생성 중입니다..."):
            response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                data = {}
            answer = data.get("answers") or data.get("answer") or "답변을 생성할 수 없습니다."
            sources = data.get("quality_metrics") or data.get("sources") or {}
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "bot", "content": answer, "sources": sources})
        else:
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "bot", "content": f"오류: {response.status_code} - {response.text}"})
    except requests.exceptions.ConnectionError:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "bot", "content": "서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요."})
    except requests.exceptions.Timeout:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "bot", "content": "서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요."})
    except Exception as e:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "bot", "content": f"알 수 없는 오류: {str(e)}"})
    set_loading(False)

# 대화 히스토리 출력 (스타일 적용)
for msg in chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🙋 사용자: {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 챗봇: {msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            st.expander("품질 정보/검색 품질").write(msg["sources"])

if st.session_state["loading"]:
    st.info("답변을 기다리는 중입니다...")
