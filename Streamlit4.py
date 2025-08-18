# custom_fab_bottom_right.py
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

# ===== CSS: pin the block after #fab-anchor to the bottom-right in a row =====
st.markdown("""
<style>
  /* pin ONLY the next Streamlit block after our anchor */
  div[data-testid="stVerticalBlock"]:has(> #fab-anchor) + div[data-testid="stVerticalBlock"]{
    position: fixed !important;
    right: 24px;
    bottom: 20px;             /* above chat input */
    z-index: 1000;
    width: auto !important;
    display: flex;            /* container itself is flex, but we still use columns for reliability */
    gap: 10px;
    align-items: center;
    pointer-events: none;     /* let typing happen under it */
  }
  /* re-enable clicks on the buttons */
  div[data-testid="stVerticalBlock"]:has(> #fab-anchor) + div[data-testid="stVerticalBlock"] button{
    pointer-events: auto;
  }
  /* button styling to match your fab look */
  div[data-testid="stVerticalBlock"]:has(> #fab-anchor) + div[data-testid="stVerticalBlock"] button {
    background:#000 !important; color:#fff !important;
    border:none !important; border-radius:9999px !important;
    padding:10px 16px !important; font-weight:600 !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
  }
  /* make the three column wrappers shrink to content */
  div[data-testid="stVerticalBlock"]:has(> #fab-anchor) + div[data-testid="stVerticalBlock"] > div {
    width: auto !important;   /* don’t stretch */
  }
</style>
""", unsafe_allow_html=True)



st.title("Underwriting Agent")

# ---------------- Render full history with custom bubbles ----------------
for m in st.session_state.messages:
    cls = "user-bubble" if m["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Fixed buttons (NO NAVIGATION, pinned at bottom) ----------------
# Place a tiny anchor; the very next Streamlit block is pinned by the CSS above.
# ===== Anchor + pinned container with HORIZONTAL COLUMNS =====
st.markdown('<span id="fab-anchor"></span>', unsafe_allow_html=True)
fab_row = st.container()
with fab_row:
    c1, c2, c3 = st.columns([1, 1, 1])   # guarantees a single row
    with c1:
        st.button("ETRAN Cheatsheet", key="fab_etran",
                  on_click=queue_question, args=("ETRAN Cheatsheet",))
    with c2:
        st.button("Valuation", key="fab_val",
                  on_click=queue_question, args=("What is the valuation?",))
    with c3:
        st.button("Goodwill value", key="fab_gw",
                  on_click=queue_question, args=("Goodwill value",))

# ---------------- Chat input ----------------
user_q = st.chat_input("Type your question here…")
if user_q:
    queue_question(user_q)

# ---------------- Answer queued question ----------------
if st.session_state.waiting_for_response and st.session_state.pending_input:
    st.markdown("<div class='assistant-bubble'>Thinking…</div>", unsafe_allow_html=True)
    answer_pending()
    st.rerun()
