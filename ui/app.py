import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
FEEDBACK_ENDPOINT = f"{API_URL}/api/v1/feedback"

st.set_page_config(
    page_title="Cedar Chatbot",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,600&family=Cinzel:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --gold: #C9A84C;
        --gold-light: #E8C97A;
        --gold-dim: rgba(201,168,76,0.18);
        --gold-border: rgba(201,168,76,0.25);
        --forest-deep: #050A06;
        --forest-dark: #0A1209;
        --forest-mid: #101A10;
        --forest-panel: #0D1610;
        --forest-card: #111F13;
        --emerald: #2A7A3B;
        --emerald-bright: #3DAA55;
        --emerald-glow: rgba(42,122,59,0.25);
        --cream: #F0E8D0;
        --cream-dim: #B8AD95;
        --muted: #6B7A5E;
        --red-accent: #9B2020;
        --white-soft: #EAE8E0;
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulseGold { 0%,100% { box-shadow: 0 0 12px rgba(201,168,76,0.15); } 50% { box-shadow: 0 0 24px rgba(201,168,76,0.3); } }
    @keyframes pulseToggle { 0%,100% { box-shadow: 4px 0 20px rgba(0,0,0,0.7), 0 0 12px rgba(201,168,76,0.3); } 50% { box-shadow: 4px 0 24px rgba(0,0,0,0.8), 0 0 20px rgba(201,168,76,0.5); } }

    .stApp {
        background: var(--forest-deep) !important;
        font-family: 'DM Sans', sans-serif !important;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(42,122,59,0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 80%, rgba(201,168,76,0.04) 0%, transparent 50%),
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") !important;
    }

    section[data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: rgba(5,10,6,0.95) !important;
        backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--gold-border) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--forest-dark) !important;
        border-right: 1px solid var(--gold-border) !important;
        background-image:
            linear-gradient(180deg, rgba(201,168,76,0.03) 0%, transparent 40%),
            url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.015' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E") !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }

    #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 6px 0 !important;
        font-family: 'DM Sans', sans-serif !important;
        animation: fadeIn 0.3s ease forwards !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14.5px !important;
        line-height: 1.7 !important;
        color: var(--cream) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(42,122,59,0.12), rgba(201,168,76,0.06)) !important;
        border: 1px solid rgba(201,168,76,0.2) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 16px 20px !important;
        position: relative !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 18px 18px 4px 18px;
        background: linear-gradient(135deg, rgba(201,168,76,0.05), transparent);
        pointer-events: none;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--forest-card) !important;
        border: 1px solid var(--gold-border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(201,168,76,0.08) !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] div[data-baseweb],
    [data-testid="stChatInput"] div[data-baseweb] > div,
    [data-testid="stChatInput"] div[data-baseweb] > div > div,
    .stChatInput,
    .stChatInput > div,
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
        background: var(--forest-deep) !important;
        border: none !important;
        border-top: 1px solid var(--gold-border) !important;
        background-image: linear-gradient(0deg, rgba(201,168,76,0.03) 0%, transparent 100%) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--forest-panel) !important;
        border: 1px solid var(--gold-border) !important;
        border-radius: 14px !important;
        color: var(--cream) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        padding: 14px 20px !important;
        caret-color: var(--gold-light) !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3), inset 0 1px 0 rgba(201,168,76,0.05) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(201,168,76,0.5) !important;
        box-shadow: 0 0 0 3px rgba(201,168,76,0.08), 0 4px 24px rgba(0,0,0,0.4) !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        font-style: italic !important;
    }

    [data-testid="stChatInputSubmitButton"] button {
        background: linear-gradient(135deg, var(--gold), #A8782A) !important;
        border: none !important;
        border-radius: 11px !important;
        color: var(--forest-deep) !important;
        box-shadow: 0 2px 16px rgba(201,168,76,0.25), inset 0 1px 0 rgba(255,255,255,0.15) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInputSubmitButton"] button:hover {
        box-shadow: 0 4px 28px rgba(201,168,76,0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: var(--forest-panel) !important;
        border: 1px solid rgba(201,168,76,0.15) !important;
        border-radius: 10px !important;
        color: var(--cream-dim) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        padding: 10px 14px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--forest-card) !important;
        border-color: rgba(201,168,76,0.35) !important;
        color: var(--gold-light) !important;
        box-shadow: 0 2px 12px rgba(201,168,76,0.1) !important;
    }

    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:active {
        border-color: rgba(201,168,76,0.3) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .new-session-btn > div > button {
        background: linear-gradient(135deg, rgba(42,122,59,0.2), rgba(201,168,76,0.1)) !important;
        border: 1px solid rgba(201,168,76,0.3) !important;
        color: var(--gold-light) !important;
        font-weight: 600 !important;
        font-family: 'Cinzel', serif !important;
        letter-spacing: 0.5px !important;
    }

    .new-session-btn > div > button:hover {
        background: linear-gradient(135deg, rgba(42,122,59,0.3), rgba(201,168,76,0.15)) !important;
        border-color: rgba(201,168,76,0.5) !important;
        box-shadow: 0 4px 20px rgba(201,168,76,0.15) !important;
    }

    .stButton > button {
        border-radius: 8px !important;
        min-height: 34px !important;
        border: 1px solid var(--gold-border) !important;
        background: var(--forest-panel) !important;
        color: var(--cream-dim) !important;
    }

    .stButton > button:hover {
        background: var(--forest-card) !important;
        border-color: rgba(201,168,76,0.35) !important;
    }

    .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    .meta-tags {
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
        margin-top: 10px;
    }

    .meta-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 10.5px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.3px;
    }

    .tag-lang { background: rgba(42,122,59,0.15); color: #5DBF74; border: 1px solid rgba(42,122,59,0.25); }
    .tag-intent { background: rgba(201,168,76,0.1); color: var(--gold-light); border: 1px solid rgba(201,168,76,0.2); }
    .tag-positive { background: rgba(42,122,59,0.12); color: #5DBF74; border: 1px solid rgba(42,122,59,0.2); }
    .tag-negative { background: rgba(155,32,32,0.12); color: #D96B6B; border: 1px solid rgba(155,32,32,0.2); }
    .tag-neutral { background: rgba(201,168,76,0.08); color: #C9A84C; border: 1px solid rgba(201,168,76,0.15); }
    .tag-time { background: rgba(255,255,255,0.04); color: var(--muted); border: 1px solid rgba(201,168,76,0.1); font-family: 'JetBrains Mono', monospace; font-size: 9.5px; }
    .tag-normalized { background: rgba(42,122,59,0.1); color: #7ECF94; border: 1px solid rgba(42,122,59,0.18); }

    .rtl-text { direction: rtl; text-align: right; font-size: 15.5px; line-height: 1.8; font-family: 'DM Sans', sans-serif; }

    .norm-box {
        background: rgba(42,122,59,0.08);
        border: 1px solid rgba(42,122,59,0.2);
        border-radius: 8px;
        padding: 10px 14px;
        color: #7ECF94;
        font-size: 13.5px;
        margin-top: 6px;
        font-family: 'DM Sans', sans-serif;
        word-break: break-word;
    }

    .welcome-block {
        text-align: center;
        padding: 50px 20px 40px;
        animation: fadeIn 0.6s ease forwards;
    }

    .welcome-shield {
        width: 120px;
        height: 120px;
        margin: 0 auto 20px;
        background: linear-gradient(135deg, #1A3320, #0D1610);
        border: 2px solid var(--gold-border);
        border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 52px;
        box-shadow: 0 0 40px rgba(201,168,76,0.15), 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(201,168,76,0.1);
        animation: pulseGold 3s ease-in-out infinite;
    }

    .welcome-title {
        font-family: 'Cinzel', serif;
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, var(--gold-light), var(--gold), #8A6020);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
        margin-bottom: 8px;
        letter-spacing: 2px;
    }

    .welcome-divider {
        width: 80px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        margin: 14px auto;
    }

    .welcome-sub {
        font-size: 14px;
        color: var(--cream-dim);
        max-width: 380px;
        margin: 0 auto;
        line-height: 1.8;
        font-family: 'Cormorant Garamond', serif;
        font-size: 16px;
        font-style: italic;
    }

    .welcome-langs {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 24px;
    }

    .welcome-lang {
        padding: 6px 18px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'Cinzel', serif;
        letter-spacing: 0.5px;
    }

    .sidebar-logo-area {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 4px 0 20px;
    }

    .sidebar-shield {
        width: 46px;
        height: 50px;
        background: linear-gradient(160deg, #1A3A20, #0A1209);
        border: 1.5px solid var(--gold-border);
        border-radius: 50% 50% 50% 50% / 55% 55% 45% 45%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 20px rgba(201,168,76,0.12), inset 0 1px 0 rgba(201,168,76,0.08);
        flex-shrink: 0;
    }

    .sidebar-title-main {
        font-family: 'Cinzel', serif;
        font-size: 16px;
        font-weight: 700;
        background: linear-gradient(135deg, var(--gold-light), var(--gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
    }

    .sidebar-title-sub {
        font-size: 10px;
        color: var(--muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-family: 'DM Sans', sans-serif;
        margin-top: 2px;
    }

    .sidebar-section-title {
        font-family: 'Cinzel', serif;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--gold);
        margin-bottom: 10px;
        opacity: 0.7;
    }

    .lang-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 13px;
        background: var(--forest-panel);
        border: 1px solid var(--gold-border);
        border-radius: 10px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }

    .lang-card:hover {
        border-color: rgba(201,168,76,0.35);
        background: var(--forest-card);
    }

    .lang-name {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--cream-dim);
        font-family: 'DM Sans', sans-serif;
    }

    .lang-example {
        font-size: 11px;
        color: var(--muted);
    }

    .stat-card {
        background: var(--forest-panel);
        border: 1px solid var(--gold-border);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
    }

    .stat-label {
        font-size: 9px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'Cinzel', serif;
        margin-bottom: 4px;
    }

    .stat-value-msgs {
        font-size: 20px;
        font-weight: 700;
        color: var(--emerald-bright);
        font-family: 'Cinzel', serif;
    }

    .stat-value-sess {
        font-size: 11px;
        font-weight: 600;
        color: var(--gold-light);
        font-family: 'JetBrains Mono', monospace;
    }

    [data-testid="stSidebar"] hr { border-color: var(--gold-border) !important; margin: 16px 0 !important; }
    hr { border-color: var(--gold-border) !important; }

    [data-testid="stExpander"] {
        background: var(--forest-card) !important;
        border: 1px solid rgba(42,122,59,0.25) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] details summary {
        padding: 10px 14px !important;
        color: #7ECF94 !important;
        font-size: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        cursor: pointer !important;
        list-style: none !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }

    [data-testid="stExpander"] details summary::-webkit-details-marker {
        display: none !important;
    }

    [data-testid="stExpander"] details summary::marker {
        display: none !important;
        content: "" !important;
    }

    [data-testid="stExpander"] svg {
        color: #7ECF94 !important;
        fill: #7ECF94 !important;
        flex-shrink: 0 !important;
        width: 14px !important;
        height: 14px !important;
    }

    [data-testid="stExpander"] span[data-testid="stExpanderToggleIcon"] {
        display: none !important;
    }

    [data-testid="stExpanderDetails"] {
        padding: 4px 14px 12px !important;
    }

    .stSpinner > div { border-top-color: var(--gold) !important; }

    [data-testid="stToast"] { background: var(--forest-card) !important; border: 1px solid var(--gold-border) !important; color: var(--cream) !important; }

    [data-testid="stAlert"] { background: rgba(155,32,32,0.08) !important; border: 1px solid rgba(155,32,32,0.2) !important; border-radius: 10px !important; color: #D96B6B !important; }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--gold-border); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(201,168,76,0.4); }

    .main .block-container {
        background: transparent !important;
        max-width: 760px !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    div[data-testid="stVerticalBlock"] { background: transparent !important; }
    .stMarkdown, .stText { color: var(--cream) !important; }

    #cedar-sidebar-toggle {
        position: fixed;
        top: 50%;
        left: 0;
        transform: translateY(-50%);
        z-index: 999999;
        width: 32px;
        height: 72px;
        background: linear-gradient(135deg, #1A3320, #0D1610);
        border: 1px solid rgba(201,168,76,0.6);
        border-left: none;
        border-radius: 0 12px 12px 0;
        color: #E8C97A;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulseToggle 2.5s ease-in-out infinite;
        transition: width 0.2s ease, background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Cinzel', serif;
        padding: 0;
    }

    #cedar-sidebar-toggle:hover {
        width: 40px;
        background: linear-gradient(135deg, #22441E, #111F13);
        border-color: rgba(201,168,76,0.9);
        color: #F0E8D0;
    }

    #cedar-sidebar-toggle.cedar-hidden {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
(function() {
    const doc = window.parent.document;

    function findStreamlitToggle() {
        const selectors = [
            '[data-testid="stSidebarCollapseButton"] button',
            '[data-testid="stSidebarCollapseButton"]',
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"]',
            '[data-testid="collapsedControl"] button',
            '[data-testid="collapsedControl"]',
            '[data-testid="baseButton-headerNoPadding"]',
            'button[kind="header"]',
            '[data-testid="stHeader"] button'
        ];
        for (const sel of selectors) {
            const el = doc.querySelector(sel);
            if (el) return el;
        }
        return null;
    }

    function isSidebarExpanded() {
        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) return false;
        const aria = sidebar.getAttribute('aria-expanded');
        if (aria !== null) return aria === 'true';
        const rect = sidebar.getBoundingClientRect();
        return rect.width > 50;
    }

    function ensureToggleButton() {
        let btn = doc.getElementById('cedar-sidebar-toggle');
        if (!btn) {
            btn = doc.createElement('button');
            btn.id = 'cedar-sidebar-toggle';
            btn.setAttribute('aria-label', 'Toggle sidebar');
            btn.innerHTML = '\\u276F';
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                const target = findStreamlitToggle();
                if (target) {
                    target.click();
                } else {
                    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) {
                        sidebar.style.display = 'block';
                        sidebar.style.transform = 'translateX(0)';
                        sidebar.setAttribute('aria-expanded', 'true');
                    }
                }
                setTimeout(updateToggleVisibility, 100);
                setTimeout(updateToggleVisibility, 400);
            });
            doc.body.appendChild(btn);
        }
        return btn;
    }

    function updateToggleVisibility() {
        const btn = ensureToggleButton();
        if (isSidebarExpanded()) {
            btn.classList.add('cedar-hidden');
        } else {
            btn.classList.remove('cedar-hidden');
        }
    }

    function init() {
        ensureToggleButton();
        updateToggleVisibility();
    }

    init();
    setTimeout(init, 200);
    setTimeout(init, 800);
    setTimeout(init, 2000);

    const obs = new MutationObserver(function() {
        updateToggleVisibility();
    });
    obs.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['aria-expanded', 'style', 'class'] });
})();
</script>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def is_arabic(text):
    return any("\u0600" <= c <= "\u06FF" for c in text)


def send_message(message):
    try:
        r = requests.post(
            CHAT_ENDPOINT,
            json={"message": message, "session_id": st.session_state.session_id},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"error": f"API error: {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is FastAPI running on port 8000?"}
    except Exception as e:
        return {"error": str(e)}


def send_feedback(message_id, rating):
    try:
        requests.post(
            FEEDBACK_ENDPOINT,
            json={"session_id": st.session_state.session_id, "message_id": message_id, "rating": rating},
            timeout=5,
        )
    except Exception:
        pass


def get_lang_label(lang):
    return {
        "english": "🇬🇧 EN",
        "arabic_msa": "🇸🇦 AR",
        "lebanese_arabic": "🇱🇧 LB",
        "lebanese_arabizi": "🇱🇧 Arabizi",
    }.get(lang, lang)


def render_tags(meta):
    tags = []
    if meta.get("detected_language"):
        tags.append(f'<span class="meta-tag tag-lang">🌐 {get_lang_label(meta["detected_language"])}</span>')
    if meta.get("intent"):
        tags.append(f'<span class="meta-tag tag-intent">◆ {meta["intent"]}</span>')
    if meta.get("sentiment"):
        s = meta["sentiment"]
        label = s.get("label", "neutral")
        emoji = {"positive": "✦", "negative": "✧", "neutral": "◇"}.get(label, "")
        tags.append(f'<span class="meta-tag tag-{label}">{emoji} {label}</span>')
    if meta.get("response_time_ms"):
        tags.append(f'<span class="meta-tag tag-time">⚡ {int(meta["response_time_ms"])}ms</span>')
    if meta.get("normalized_text"):
        tags.append('<span class="meta-tag tag-normalized">✦ Normalized</span>')
    return f'<div class="meta-tags">{"".join(tags)}</div>'


def render_normalized_text(normalized_text):
    if not normalized_text:
        return
    with st.expander("Normalized Text"):
        st.markdown(
            f'<div class="norm-box">{normalized_text}</div>',
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-area">
        <div class="sidebar-shield">🌲</div>
        <div>
            <div class="sidebar-title-main">Cedar</div>
            <div class="sidebar-title-sub">Trilingual AI Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.4),transparent);margin-bottom:16px;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Supported Languages</div>', unsafe_allow_html=True)

    st.markdown("""
    <div>
        <div class="lang-card">
            <span style="font-size:20px;">🇬🇧</span>
            <div>
                <div class="lang-name">English</div>
                <div class="lang-example">Hello, how are you?</div>
            </div>
        </div>
        <div class="lang-card">
            <span style="font-size:20px;">🇸🇦</span>
            <div>
                <div class="lang-name">Arabic (MSA)</div>
                <div class="lang-example" style="direction:rtl;">كيف حالك؟</div>
            </div>
        </div>
        <div class="lang-card">
            <span style="font-size:20px;">🇱🇧</span>
            <div>
                <div class="lang-name">Lebanese Arabizi</div>
                <div class="lang-example">keefak ya zalame?</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Try These</div>', unsafe_allow_html=True)

    for ex in [
        "Hello! What is NLP?", "keefak ya zalame?", "shu 3am ta3mel?",
        "7abibi kifak lyom?", "كيف حالك؟", "اشرح لي التعلم العميق",
        "Tell me about Lebanon", "yalla ne7ke 3an el AI",
    ]:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_message = ex
            st.rerun()

    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:16px 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Session</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">Messages</div>'
            f'<div class="stat-value-msgs">{len(st.session_state.messages)}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">Session</div>'
            f'<div class="stat-value-sess">{st.session_state.session_id[:6]}...</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="new-session-btn">', unsafe_allow_html=True)
    if st.button("✦ New Session", use_container_width=True, key="new_session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-block">
        <div class="welcome-shield">🌲</div>
        <div class="welcome-title">Cedar Chatbot</div>
        <div class="welcome-divider"></div>
        <div class="welcome-sub">A trilingual AI assistant that understands English, Arabic, and Lebanese dialect — including Arabizi.</div>
        <div class="welcome-langs">
            <span class="welcome-lang" style="background:rgba(42,122,59,0.15);color:#5DBF74;border:1px solid rgba(42,122,59,0.25);">English</span>
            <span class="welcome-lang" style="background:rgba(201,168,76,0.1);color:#C9A84C;border:1px solid rgba(201,168,76,0.2);">العربية</span>
            <span class="welcome-lang" style="background:rgba(155,32,32,0.1);color:#D96B6B;border:1px solid rgba(155,32,32,0.2);">Arabizi</span>
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
            render_normalized_text(meta.get("normalized_text"))
            msg_id = msg.get("message_id", "")
            if msg_id:
                c1, c2, c3 = st.columns([1, 1, 8])
                with c1:
                    if st.button("👍", key=f"up_{msg_id}"):
                        send_feedback(msg_id, 1)
                        st.toast("Noted — thank you ✦")
                with c2:
                    if st.button("👎", key=f"dn_{msg_id}"):
                        send_feedback(msg_id, -1)
                        st.toast("Understood. We shall improve.")


def process_message(text):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user", avatar="🧑"):
        if is_arabic(text):
            st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
        else:
            st.write(text)

    with st.chat_message("assistant", avatar="🌲"):
        with st.spinner("Cedar is thinking..."):
            result = send_message(text)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["error"], "metadata": {}}
            )
        else:
            response_text = result.get("response", "")
            if is_arabic(response_text):
                st.markdown(f'<div class="rtl-text">{response_text}</div>', unsafe_allow_html=True)
            else:
                st.write(response_text)

            metadata = result.get("metadata", {})
            st.markdown(render_tags(metadata), unsafe_allow_html=True)
            render_normalized_text(metadata.get("normalized_text"))

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "message_id": result.get("message_id", ""),
                "metadata": metadata,
            })
    st.rerun()


if "pending_message" in st.session_state:
    text = st.session_state.pending_message
    del st.session_state.pending_message
    process_message(text)

if prompt := st.chat_input("Speak your message... (EN / AR / Arabizi)"):
    process_message(prompt)