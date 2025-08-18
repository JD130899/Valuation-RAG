# custom_fab_no_refresh.py
import time
import streamlit as st

st.set_page_config(page_title="Underwriting Agent", layout="wide")

# ---------- Global store so history survives reloads ----------
@st.cache_resource
def _store():
    return {"messages": []}

store = _store()

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = store["messages"] or [
        {"role": "assistant", "content": "Hi! Ask anything about your valuation report"},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "next_id" not in st.session_state:
    st.session_state.next_id = 0

def _new_id():
    st.session_state.next_id += 1
    return f"m{st.session_state.next_id}"

def _sync_store():
    store["messages"] = st.session_state.messages

def queue_question(q: str):
    if not q:
        return
    st.session_state.pending_input = q
    st.session_state.waiting_for_response = True
    st.session_state.messages.append({"id": _new_id(), "role": "user", "content": q})
    _sync_store()

def answer_pending():
    time.sleep(0.2)  # simulate LLM
    q = st.session_state.pending_input
    ans = f"(demo) You asked: {q}"
    st.session_state.messages.append({"id": _new_id(), "role": "assistant", "content": ans})
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    _sync_store()

# --------------- Styles ---------------
st.markdown("""
<style>
  .block-container { padding-bottom: 140px; }

  .fab-wrap {
    position: fixed;
    right: 24px;
    bottom: 110px;
    z-index: 1000;
    display: flex; gap: 10px; align-items: center; justify-content: flex-end;
  }
  .fab-btn {
    background: #000 !important; color: #fff !important;
    border-radius: 9999px !important; padding: 10px 16px !important; border: none !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25); font-weight: 600; cursor: pointer;
  }
  .fab-btn:hover { filter: brightness(1.08); }

  /* custom chat bubbles */
  .user-bubble {
    background: #007bff; color: #fff;
    padding: 8px 12px; border-radius: 12px;
    margin: 4px 0; max-width: 60%;
    float: right; clear: both;
  }
  .assistant-bubble {
    background: #1e1e1e; color: #fff;
    padding: 8px 12px; border-radius: 12px;
    margin: 4px 0; max-width: 60%;
    float: left; clear: both;
  }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📑 Underwriting Agent")
    st.selectbox("Select a PDF from Google Drive", ["JB Tile & Stone Certified Valuation", "Another PDF"])
    st.button("Load Selected PDF")
    st.button("➕ New chat")

# ---------------- Main title ----------------
st.title("Underwriting Agent")

# ---------------- Render full history ----------------
for m in st.session_state.messages:
    cls = "user-bubble" if m["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Fixed buttons (NO NAVIGATION) ----------------
fab1 = st.button("ETRAN Cheatsheet", key="etran", help="Ask about ETRAN", use_container_width=False)
fab2 = st.button("Valuation", key="valuation", help="Ask about valuation", use_container_width=False)
fab3 = st.button("Goodwill value", key="goodwill", help="Ask about goodwill", use_container_width=False)

# Place them using CSS
st.markdown("""
<style>
div[data-testid="stButton"] {display:inline-block;}
</style>
<div class="fab-wrap"></div>
""", unsafe_allow_html=True)

# Check which button pressed
if fab1:
    queue_question("ETRAN Cheatsheet")
if fab2:
    queue_question("What is the valuation?")
if fab3:
    queue_question("Goodwill value")

# ---------------- Chat input ----------------
user_q = st.chat_input("Type your question here…")
if user_q:
    queue_question(user_q)

# ---------------- Answer queued question ----------------
if st.session_state.waiting_for_response and st.session_state.pending_input:
    st.markdown("<div class='assistant-bubble'>Thinking…</div>", unsafe_allow_html=True)
    answer_pending()
    st.rerun()
