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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .stApp {
        background: #08090d !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    section[data-testid="stMain"] {
        background: #08090d !important;
    }

    [data-testid="stHeader"] {
        background: rgba(8,9,13,.9) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid #252840 !important;
    }

    [data-testid="stSidebar"] {
        background: #0e1017 !important;
        border-right: 1px solid #252840 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #0e1017 !important;
    }

    /* ── Hide defaults ── */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 8px 0 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14.5px !important;
        line-height: 1.65 !important;
        color: #eaedf6 !important;
    }

    /* User messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(91,141,239,.08), rgba(99,102,241,.05)) !important;
        border: 1px solid rgba(91,141,239,.12) !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 14px 18px !important;
    }

    /* Bot messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #13151e !important;
        border: 1px solid #252840 !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 14px 18px !important;
    }

    /* ── Chat input — kill ALL white borders ── */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] div[data-baseweb],
    [data-testid="stChatInput"] div[data-baseweb] > div,
    [data-testid="stChatInput"] div[data-baseweb] > div > div,
    .stChatInput,
    .stChatInput > div,
    .stChatInput > div > div,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="textarea"] > div > div {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        background: #08090d !important;
        border: none !important;
        border-top: 1px solid #252840 !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #13151e !important;
        border: 1px solid #252840 !important;
        border-radius: 14px !important;
        color: #eaedf6 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 14px 18px !important;
        caret-color: #2dd4a0 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(45,212,160,.3) !important;
        box-shadow: 0 0 0 3px rgba(45,212,160,.06) !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6c7296 !important;
    }

    [data-testid="stChatInputSubmitButton"] button {
        background: linear-gradient(135deg, #2dd4a0, #0ea5e9) !important;
        border: none !important;
        border-radius: 11px !important;
        color: #fff !important;
        box-shadow: 0 2px 12px rgba(45,212,160,.2) !important;
    }

    [data-testid="stChatInputSubmitButton"] button:hover {
        box-shadow: 0 4px 20px rgba(45,212,160,.3) !important;
    }

    /* ── Sidebar buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        background: #13151e !important;
        border: 1px solid #252840 !important;
        border-radius: 10px !important;
        color: #b0b5cc !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        transition: all .15s ease !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #191c28 !important;
        border-color: #2e3250 !important;
        color: #eaedf6 !important;
    }

    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:active {
        border-color: #2e3250 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* New session button */
    .new-session-btn > div > button {
        background: linear-gradient(135deg, rgba(45,212,160,.1), rgba(14,165,233,.06)) !important;
        border: 1px solid rgba(45,212,160,.2) !important;
        color: #2dd4a0 !important;
        font-weight: 600 !important;
    }

    .new-session-btn > div > button:hover {
        background: linear-gradient(135deg, rgba(45,212,160,.16), rgba(14,165,233,.1)) !important;
        border-color: rgba(45,212,160,.35) !important;
    }

    /* Feedback buttons */
    .stButton > button {
        border-radius: 8px !important;
        min-height: 34px !important;
        border: 1px solid #252840 !important;
        background: #191c28 !important;
        color: #b0b5cc !important;
    }

    .stButton > button:hover {
        background: #1f2233 !important;
        border-color: #2e3250 !important;
    }

    .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    /* ── Tags ── */
    .meta-tags {
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
        margin-top: 8px;
    }

    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
    }

    .tag-lang { background: rgba(91,141,239,.1); color: #5b8def; border: 1px solid rgba(91,141,239,.15); }
    .tag-intent { background: rgba(167,139,250,.1); color: #a78bfa; border: 1px solid rgba(167,139,250,.15); }
    .tag-positive { background: rgba(45,212,160,.1); color: #2dd4a0; border: 1px solid rgba(45,212,160,.15); }
    .tag-negative { background: rgba(248,113,113,.1); color: #f87171; border: 1px solid rgba(248,113,113,.15); }
    .tag-neutral { background: rgba(251,191,36,.1); color: #fbbf24; border: 1px solid rgba(251,191,36,.15); }
    .tag-time { background: rgba(255,255,255,.04); color: #6c7296; border: 1px solid #252840; font-family: 'JetBrains Mono', monospace; font-size: 10px; }
    .tag-normalized { background: rgba(34,211,238,.1); color: #22d3ee; border: 1px solid rgba(34,211,238,.15); }

    /* ── RTL ── */
    .rtl-text { direction: rtl; text-align: right; font-size: 15.5px; line-height: 1.7; }

    /* ── Normalized text ── */
    .norm-box {
        background: rgba(34,211,238,.06);
        border: 1px solid rgba(34,211,238,.12);
        border-radius: 8px;
        padding: 10px 14px;
        direction: rtl;
        text-align: right;
        color: #22d3ee;
        font-size: 14px;
        margin-top: 6px;
    }

    /* ── Welcome ── */
    .welcome-block { text-align: center; padding: 60px 20px 40px; }
    .welcome-icon { font-size: 56px; margin-bottom: 12px; filter: drop-shadow(0 0 24px rgba(45,212,160,.2)); }
    .welcome-title { font-family: 'DM Sans', sans-serif; font-size: 26px; font-weight: 700; color: #eaedf6; margin-bottom: 6px; letter-spacing: -0.5px; }
    .welcome-sub { font-size: 14px; color: #6c7296; max-width: 400px; margin: 0 auto; line-height: 1.7; }
    .welcome-langs { display: flex; justify-content: center; gap: 10px; margin-top: 20px; }
    .welcome-lang { padding: 6px 16px; border-radius: 8px; font-size: 12.5px; font-weight: 600; font-family: 'DM Sans', sans-serif; }

    /* ── Sidebar ── */
    .sidebar-section-title { font-family: 'DM Sans', sans-serif; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #6c7296; margin-bottom: 10px; }

    [data-testid="stSidebar"] hr { border-color: #252840 !important; margin: 16px 0 !important; }
    hr { border-color: #252840 !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] { background: transparent !important; border: 1px solid rgba(34,211,238,.12) !important; border-radius: 8px !important; }
    [data-testid="stExpander"] summary { color: #22d3ee !important; font-size: 12px !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #2dd4a0 !important; }

    /* ── Toast ── */
    [data-testid="stToast"] { background: #13151e !important; border: 1px solid #252840 !important; color: #eaedf6 !important; }

    /* ── Error ── */
    [data-testid="stAlert"] { background: rgba(248,113,113,.06) !important; border: 1px solid rgba(248,113,113,.15) !important; border-radius: 10px !important; color: #f87171 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #252840; border-radius: 10px; }

    /* ── Kill all remaining whites ── */
    .main .block-container { background: #08090d !important; }
    div[data-testid="stVerticalBlock"] { background: transparent !important; }
    .stMarkdown, .stText { color: #eaedf6 !important; }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def is_arabic(text):
    return any("\u0600" <= c <= "\u06FF" for c in text)

def send_message(message):
    try:
        r = requests.post(CHAT_ENDPOINT, json={"message": message, "session_id": st.session_state.session_id}, timeout=60)
        return r.json() if r.status_code == 200 else {"error": f"API error: {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is FastAPI running on port 8000?"}
    except Exception as e:
        return {"error": str(e)}

def send_feedback(message_id, rating):
    try:
        requests.post(FEEDBACK_ENDPOINT, json={"session_id": st.session_state.session_id, "message_id": message_id, "rating": rating}, timeout=5)
    except Exception:
        pass

def get_lang_label(lang):
    return {"english": "🇬🇧 EN", "arabic_msa": "🇸🇦 AR", "lebanese_arabic": "🇱🇧 LB", "lebanese_arabizi": "🇱🇧 Arabizi"}.get(lang, lang)

def render_tags(meta):
    tags = []
    if meta.get("detected_language"):
        tags.append(f'<span class="meta-tag tag-lang">🌐 {get_lang_label(meta["detected_language"])}</span>')
    if meta.get("intent"):
        tags.append(f'<span class="meta-tag tag-intent">🎯 {meta["intent"]}</span>')
    if meta.get("sentiment"):
        s = meta["sentiment"]
        label = s.get("label", "neutral")
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(label, "")
        tags.append(f'<span class="meta-tag tag-{label}">{emoji} {label}</span>')
    if meta.get("response_time_ms"):
        tags.append(f'<span class="meta-tag tag-time">⚡ {int(meta["response_time_ms"])}ms</span>')
    if meta.get("normalized_text"):
        tags.append(f'<span class="meta-tag tag-normalized">📝 Normalized</span>')
    return f'<div class="meta-tags">{"".join(tags)}</div>'

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 0 16px;">
        <div style="width:42px;height:42px;background:linear-gradient(135deg,#2dd4a0,#0ea5e9);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 24px rgba(45,212,160,.2);">🌲</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#eaedf6;letter-spacing:-.3px;">Cedar Chatbot</div>
            <div style="font-size:11px;color:#6c7296;">Trilingual AI Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Supported Languages</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:4px;">
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:#13151e;border:1px solid #252840;border-radius:10px;">
            <span style="font-size:20px;">🇬🇧</span>
            <div><div style="font-size:12.5px;font-weight:600;color:#b0b5cc;">English</div><div style="font-size:11px;color:#6c7296;">Hello, how are you?</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:#13151e;border:1px solid #252840;border-radius:10px;">
            <span style="font-size:20px;">🇸🇦</span>
            <div><div style="font-size:12.5px;font-weight:600;color:#b0b5cc;">Arabic (MSA)</div><div style="font-size:11px;color:#6c7296;direction:rtl;">كيف حالك؟</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:#13151e;border:1px solid #252840;border-radius:10px;">
            <span style="font-size:20px;">🇱🇧</span>
            <div><div style="font-size:12.5px;font-weight:600;color:#b0b5cc;">Lebanese Arabizi</div><div style="font-size:11px;color:#6c7296;">keefak ya zalame?</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Try These</div>', unsafe_allow_html=True)

    for ex in ["Hello! What is NLP?", "keefak ya zalame?", "shu 3am ta3mel?", "7abibi kifak lyom?", "كيف حالك؟", "اشرح لي التعلم العميق", "Tell me about Lebanon", "yalla ne7ke 3an el AI"]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_message = ex
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="background:#13151e;border:1px solid #252840;border-radius:8px;padding:10px 12px;text-align:center;"><div style="font-size:10px;color:#6c7296;margin-bottom:2px;">Messages</div><div style="font-size:18px;font-weight:700;color:#5b8def;">{len(st.session_state.messages)}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="background:#13151e;border:1px solid #252840;border-radius:8px;padding:10px 12px;text-align:center;"><div style="font-size:10px;color:#6c7296;margin-bottom:2px;">Session</div><div style="font-size:11px;font-weight:600;color:#2dd4a0;font-family:monospace;">{st.session_state.session_id[:6]}...</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="new-session-btn">', unsafe_allow_html=True)
    if st.button("✨ New Session", use_container_width=True, key="new_session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-block">
        <div class="welcome-icon">🌲</div>
        <div class="welcome-title">Welcome to Cedar</div>
        <div class="welcome-sub">A trilingual AI chatbot that understands English, Arabic, and Lebanese dialect — including Arabizi.</div>
        <div class="welcome-langs">
            <span class="welcome-lang" style="background:rgba(91,141,239,.1);color:#5b8def;">English</span>
            <span class="welcome-lang" style="background:rgba(45,212,160,.1);color:#2dd4a0;">العربية</span>
            <span class="welcome-lang" style="background:rgba(167,139,250,.1);color:#a78bfa;">Arabizi</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌲"):
        if is_arabic(msg["content"]):
            st.markdown(f'<div class="rtl-text">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.write(msg["content"])

        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            st.markdown(render_tags(meta), unsafe_allow_html=True)
            if meta.get("normalized_text"):
                with st.expander("📝 Normalized Text"):
                    st.markdown(f'<div class="norm-box">{meta["normalized_text"]}</div>', unsafe_allow_html=True)
            msg_id = msg.get("message_id", "")
            if msg_id:
                c1, c2, c3 = st.columns([1, 1, 8])
                with c1:
                    if st.button("👍", key=f"up_{msg_id}"):
                        send_feedback(msg_id, 1)
                        st.toast("Thanks! Feedback recorded 🌲")
                with c2:
                    if st.button("👎", key=f"dn_{msg_id}"):
                        send_feedback(msg_id, -1)
                        st.toast("Got it. We'll improve!")


def process_message(text):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user", avatar="🧑"):
        if is_arabic(text):
            st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
        else:
            st.write(text)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("🌲 Thinking..."):
            result = send_message(text)
        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "content": result["error"], "metadata": {}})
        else:
            response_text = result.get("response", "")
            if is_arabic(response_text):
                st.markdown(f'<div class="rtl-text">{response_text}</div>', unsafe_allow_html=True)
            else:
                st.write(response_text)
            metadata = result.get("metadata", {})
            st.markdown(render_tags(metadata), unsafe_allow_html=True)
            if metadata.get("normalized_text"):
                with st.expander("📝 Normalized Text"):
                    st.markdown(f'<div class="norm-box">{metadata["normalized_text"]}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "message_id": result.get("message_id", ""), "metadata": metadata})
    st.rerun()

if "pending_message" in st.session_state:
    text = st.session_state.pending_message
    del st.session_state.pending_message
    process_message(text)

if prompt := st.chat_input("Type a message... (EN / AR / Arabizi)"):
    process_message(prompt)