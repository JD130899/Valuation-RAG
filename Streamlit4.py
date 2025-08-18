import os
import streamlit as st
import streamlit.components.v1 as components

# ---------------------- Page + Styles ----------------------
st.set_page_config(page_title="Pinned Buttons Chat", layout="wide")

st.markdown("""
<style>
  /* Give some breathing room below so the pinned toolbar doesn't cover content */
  .block-container { padding-bottom: 160px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Chat State ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_user_text" not in st.session_state:
    st.session_state.pending_user_text = None

# ---------------------- LLM Call ----------------------
def call_llm(prompt: str) -> str:
    """Replace with your own model/provider if you like."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error) {e}"

# ---------------------- Helpers ----------------------
def send_user(text: str):
    if not text:
        return
    st.session_state.messages.append({"role": "user", "content": text})
    reply = call_llm(text)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------- Pinned Toolbar (Bottom-Right) ----------------------
# Put the sentinel in its own container so we can grab the wrapper div and pin it.
toolbar = st.container()
with toolbar:
    st.markdown("<span id='pin-bottom-right'></span>", unsafe_allow_html=True)
    # the actual buttons (you can change labels/actions as you wish)
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Valuation", key="btn_valuation"):
            st.session_state.pending_user_text = "Valuation"
    with c2:
        if st.button("Good will", key="btn_goodwill"):
            st.session_state.pending_user_text = "Good will"

# JS to pin that wrapper above the chat input (bottom-right)
components.html("""
<script>
(function pin(){
  const d = window.parent.document;
  const sentinel = d.querySelector('#pin-bottom-right');
  if (!sentinel) return setTimeout(pin, 120);
  const block = sentinel.closest('div[data-testid="stVerticalBlock"]');
  if (!block) return setTimeout(pin, 120);

  Object.assign(block.style, {
    position: 'fixed',
    right: '24px',
    bottom: '88px',  // sits above Streamlit's chat input bar
    zIndex: '10000',
    display: 'flex',
    gap: '10px',
    padding: '8px 12px',
    borderRadius: '9999px',
    background: 'rgba(17,24,39,.96)',
    border: '1px solid rgba(255,255,255,.12)',
    boxShadow: '0 8px 28px rgba(0,0,0,.35)'
  });

  // keep children from stretching
  Array.from(block.children).forEach(ch => ch.style.width = 'auto');
})();
</script>
""", height=0)

# If a pinned button was clicked, send it as a user message now
if st.session_state.pending_user_text:
    send_user(st.session_state.pending_user_text)
    st.session_state.pending_user_text = None

# ---------------------- Chat Display ----------------------
st.title("Chat with Pinned Bottom-Right Buttons")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------------- Chat Input ----------------------
user_inp = st.chat_input("Type your question…")
if user_inp:
    send_user(user_inp)
    # Re-render last two messages immediately for responsiveness
    with st.chat_message("user"):
        st.markdown(user_inp)
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[-1]["content"])
