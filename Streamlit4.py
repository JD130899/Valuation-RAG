# custom_fab_bottom_right.py
import os
import time
import streamlit as st
from dotenv import load_dotenv
from gdrive_utils import get_drive_service, get_all_pdfs, download_pdf

# ================= Setup =================
load_dotenv()
st.set_page_config(page_title="Underwriting Agent (Demo)", layout="wide")

# ---------- Global store so history survives reloads ----------
@st.cache_resource
def _store():
    return {"messages": []}
store = _store()

# ---------- helpers (reference-style welcome) ----------
def _reset_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."},
        {"role": "assistant", "content": "What can I help you with?"}
    ]
    st.session_state.pending_input = None
    st.session_state.waiting_for_response = False
    store["messages"] = st.session_state.messages  # keep cache in sync

def _sync_store():
    store["messages"] = st.session_state.messages

def _new_id():
    st.session_state.next_id += 1
    return f"m{st.session_state.next_id}"

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

# ---------------- Session state ----------------
if "_initialized" not in st.session_state:
    st.session_state._initialized = True
    st.session_state.next_id = 0
    # start with clean two-line welcome so old qs don't show
    _reset_chat()

st.session_state.setdefault("last_synced_file_id", None)
st.session_state.setdefault("uploaded_file_from_drive", None)
st.session_state.setdefault("uploaded_file_name", None)

# --------------- Styles ---------------
st.markdown("""
<style>
  .block-container { padding-bottom: 140px; }

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
  .clearfix::after {content:"";display:table;clear:both;}

  /* ===== Fixed bottom-right Streamlit buttons (no page reload) ===== */
  #fab-sentinel { height: 0; display: block; }
  /* The horizontal block that comes right after our sentinel */
  div[data-testid="stVerticalBlock"]:has(> #fab-sentinel) + div[data-testid="stHorizontalBlock"]{
    position: fixed !important;
    right: 24px;
    bottom: 88px;      /* sits above st.chat_input */
    z-index: 1000;
    display: flex; gap: 10px;
    width: auto !important;
  }
  /* shrink children so buttons size to content */
  div[data-testid="stVerticalBlock"]:has(> #fab-sentinel) + div[data-testid="stHorizontalBlock"] > div{
    width: auto !important;
  }
  /* button styling */
  div[data-testid="stVerticalBlock"]:has(> #fab-sentinel) + div[data-testid="stHorizontalBlock"] button {
    background:#000 !important; color:#fff !important;
    border:none !important; border-radius:9999px !important;
    padding:10px 18px !important; font-weight:600 !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
  }
</style>
""", unsafe_allow_html=True)

# ================= Sidebar: Google Drive loader (reference UX) =================
service = get_drive_service()
pdf_files = get_all_pdfs(service)

st.sidebar.title("Underwriting Agent")
if pdf_files:
    names = [f["name"] for f in pdf_files]
    sel = st.sidebar.selectbox("📂 Select a PDF from Google Drive", names, index=0 if names else None)
    chosen = next((f for f in pdf_files if f["name"] == sel), None)
    if st.sidebar.button("📥 Load Selected PDF"):
        if chosen:
            fid, fname = chosen["id"], chosen["name"]
            if fid == st.session_state.last_synced_file_id:
                st.sidebar.info("✅ Already loaded.")
            else:
                path = download_pdf(service, fid, fname)
                if path:
                    with open(path, "rb") as fh:
                        st.session_state.uploaded_file_from_drive = fh.read()
                    st.session_state.uploaded_file_name = fname
                    st.session_state.last_synced_file_id = fid
                    _reset_chat()
else:
    st.sidebar.warning("📭 No PDFs found in Drive.")

# Optional: manual clear (handy while testing)
if st.sidebar.button("🧹 New chat"):
    _reset_chat()

# ================= Main =================
st.title("Underwriting Agent (Demo)")

if st.session_state.uploaded_file_name:
    st.info(f"Using file: **{st.session_state.uploaded_file_name}**")

# ---------------- Render full history with custom bubbles ----------------
for m in st.session_state.messages:
    cls = "user-bubble" if m["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{cls} clearfix'>{m['content']}</div>", unsafe_allow_html=True)

# ---------------- Fixed buttons (Streamlit, not HTML forms) ----------------
# Sentinel + the very next st.columns becomes the fixed bar via CSS above
st.markdown('<span id="fab-sentinel"></span>', unsafe_allow_html=True)
b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    st.button("ETRAN Cheatsheet", key="fab_etran",
              on_click=queue_question, args=("ETRAN Cheatsheet",))
with b2:
    st.button("Valuation", key="fab_val",
              on_click=queue_question, args=("What is the valuation?",))
with b3:
    st.button("Goodwill value", key="fab_gw",
              on_click=queue_question, args=("Goodwill value",))

# ---------------- Chat input ----------------
user_q = st.chat_input("Type your question here…")
if user_q:
    queue_question(user_q)

# ---------------- Answer queued question ----------------
if st.session_state.waiting_for_response and st.session_state.pending_input:
    st.markdown("<div class='assistant-bubble clearfix'>Thinking…</div>", unsafe_allow_html=True)
    answer_pending()
    st.rerun()
