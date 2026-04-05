import uuid
import requests
import streamlit as st


API_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
FEEDBACK_ENDPOINT = f"{API_URL}/api/v1/feedback"


st.set_page_config(
    page_title="🌲 Cedar Chatbot",
    page_icon="🌲",
    layout="centered",
    initial_sidebar_state="expanded",
)



st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    .user-message {
        background: linear-gradient(135deg, #065f46, #064e3b);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e4e4e7;
        max-width: 85%;
        margin-left: auto;
    }
    .bot-message {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e4e4e7;
        max-width: 85%;
    }

    .rtl { direction: rtl; text-align: right; }

    .meta-badge {
        display: inline-block;
        background: rgba(255,255,255,0.06);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        color: #9ca3af;
        margin-right: 4px;
        margin-top: 6px;
    }

    .sidebar-stat {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .sidebar-stat .label { color: #9ca3af; font-size: 0.8rem; }
    .sidebar-stat .value { color: #10b981; font-size: 1.2rem; font-weight: 700; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {"total_messages": 0, "languages": set()}

with st.sidebar:
    st.markdown("## 🌲 Cedar Chatbot")
    st.markdown("---")

    st.markdown("### Supported Languages")
    st.markdown("🇬🇧 **English** — Hello, how are you?")
    st.markdown("🇸🇦 **Arabic** — كيف حالك؟")
    st.markdown("🇱🇧 **Lebanese** — keefak ya zalame?")

    st.markdown("---")

    st.markdown("### Session Info")
    st.markdown(f"**ID:** `{st.session_state.session_id[:8]}...`")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    st.markdown("---")

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.stats = {"total_messages": 0, "languages": set()}
        st.rerun()

    st.markdown("---")
    st.markdown("### Try These")
    examples = [
        "Hello! What is NLP?",
        "keefak ya zalame?",
        "shu 3am ta3mel?",
        "كيف حالك؟",
        "اشرح لي التعلم العميق",
        "7abibi kifak lyom?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_message = ex
            st.rerun()


def is_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def send_message(message: str) -> dict:
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={
                "message": message,
                "session_id": st.session_state.session_id,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def send_feedback(message_id: str, rating: int):
    try:
        requests.post(
            FEEDBACK_ENDPOINT,
            json={
                "session_id": st.session_state.session_id,
                "message_id": message_id,
                "rating": rating,
            },
            timeout=5,
        )
    except Exception:
        pass


def render_metadata(metadata: dict):
    badges = []
    if metadata.get("detected_language"):
        lang_map = {
            "english": "🇬🇧 EN",
            "arabic_msa": "🇸🇦 AR",
            "lebanese_arabic": "🇱🇧 LB-AR",
            "lebanese_arabizi": "🇱🇧 Arabizi",
        }
        lang = lang_map.get(metadata["detected_language"], metadata["detected_language"])
        badges.append(f"🌐 {lang}")

    if metadata.get("intent"):
        badges.append(f"🎯 {metadata['intent']}")

    if metadata.get("sentiment"):
        sent = metadata["sentiment"]
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sent.get("label", ""), "")
        badges.append(f"{emoji} {sent.get('label', '')}")

    if metadata.get("normalized_text"):
        badges.append(f"📝 Normalized")

    if metadata.get("response_time_ms"):
        badges.append(f"⚡ {metadata['response_time_ms']}ms")

    return " ".join(f'<span class="meta-badge">{b}</span>' for b in badges)

st.markdown("# 🌲 Cedar Chatbot")
st.markdown("*Trilingual AI — English · Arabic · Lebanese*")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌲"):
        if is_arabic(msg["content"]):
            st.markdown(f'<div class="rtl">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(msg["content"])

        if msg["role"] == "assistant" and "metadata" in msg:
            st.markdown(render_metadata(msg["metadata"]), unsafe_allow_html=True)

            meta = msg.get("metadata", {})
            if meta.get("normalized_text"):
                with st.expander("📝 Normalized Text"):
                    st.markdown(f'<div class="rtl">{meta["normalized_text"]}</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 6])
            msg_id = msg.get("message_id", "")
            with col1:
                if st.button("👍", key=f"up_{msg_id}"):
                    send_feedback(msg_id, 1)
                    st.toast("Thanks for the feedback! 🌲")
            with col2:
                if st.button("👎", key=f"dn_{msg_id}"):
                    send_feedback(msg_id, -1)
                    st.toast("Feedback recorded. We'll improve!")

if "pending_message" in st.session_state:
    pending = st.session_state.pending_message
    del st.session_state.pending_message

    st.session_state.messages.append({"role": "user", "content": pending})

    with st.chat_message("user", avatar="🧑"):
        st.write(pending)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Thinking..."):
            result = send_message(pending)

        if "error" in result:
            st.error(result["error"])
        else:
            response_text = result.get("response", "")
            if is_arabic(response_text):
                st.markdown(f'<div class="rtl">{response_text}</div>', unsafe_allow_html=True)
            else:
                st.write(response_text)

            metadata = result.get("metadata", {})
            st.markdown(render_metadata(metadata), unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "message_id": result.get("message_id", ""),
                "metadata": metadata,
            })

    st.rerun()

if prompt := st.chat_input("Type a message... (EN / AR / Arabizi)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        if is_arabic(prompt):
            st.markdown(f'<div class="rtl">{prompt}</div>', unsafe_allow_html=True)
        else:
            st.write(prompt)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("🌲 Processing..."):
            result = send_message(prompt)

        if "error" in result:
            st.error(result["error"])
        else:
            response_text = result.get("response", "")
            if is_arabic(response_text):
                st.markdown(f'<div class="rtl">{response_text}</div>', unsafe_allow_html=True)
            else:
                st.write(response_text)

            metadata = result.get("metadata", {})
            st.markdown(render_metadata(metadata), unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "message_id": result.get("message_id", ""),
                "metadata": metadata,
            })

    st.rerun()