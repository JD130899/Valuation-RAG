import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Pinned Buttons Chat", layout="wide")
st.markdown("<style>.block-container{padding-bottom:160px!important}</style>", unsafe_allow_html=True)

# ---------------- state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []        # [{"role":"user"/"assistant","content":str}]
if "quick_action" not in st.session_state:
    st.session_state.quick_action = None  # holds button text for this run only

# ---------------- model call ----------------
def call_llm(prompt: str) -> str:
    # Replace with your backend if you like
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM error) {e}"

# ---------------- pinned bottom-right actions (compact) ----------------
pill = st.container()
with pill:
    st.markdown("<span id='pin-bottom-right'></span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Valuation", key="qa_val"):
            st.session_state.quick_action = "Valuation"
    with c2:
        if st.button("Good will", key="qa_gw"):
            st.session_state.quick_action = "Good will"

# compact pill styling (no full-width bar)
components.html("""
<script>
(function pin(){
  const d = window.parent.document;
  const mark = d.querySelector('#pin-bottom-right');
  if(!mark) return setTimeout(pin,120);
  const block = mark.closest('div[data-testid="stVerticalBlock"]');
  if(!block) return setTimeout(pin,120);
  if(block.dataset.pinned==="1") return;
  block.dataset.pinned="1";
  Object.assign(block.style,{
    position:'fixed',right:'18px',bottom:'88px',zIndex:'10000',
    display:'inline-flex',gap:'8px',padding:'6px 8px',
    borderRadius:'9999px',background:'rgba(17,24,39,.96)',
    border:'1px solid rgba(255,255,255,.12)',boxShadow:'0 8px 28px rgba(0,0,0,.35)',
    width:'fit-content',maxWidth:'none'
  });
  Array.from(block.children||[]).forEach(ch=>{ch.style.width='auto';ch.style.margin='0'});
  block.querySelectorAll('button').forEach(b=>{b.style.padding='6px 12px'; b.style.borderRadius='9999px';});
})();
</script>
""", height=0)

st.title("Chat with Pinned Bottom-Right Buttons")

# ---------------- collect user input (do NOT render bubbles yet) ----------------
typed = st.chat_input("Type your question…")

# determine the user text for this run (buttons win if both happen)
user_text = st.session_state.quick_action or typed
st.session_state.quick_action = None  # consume the quick action

# ---------------- handle input ONCE, update history ONCE ----------------
thinking_slot = st.empty()  # non-persisted area to avoid flicker
if user_text:
    # append user to history
    st.session_state.messages.append({"role": "user", "content": user_text})
    # show transient "Thinking..." under the chat title while we wait
    with thinking_slot.container():
        with st.chat_message("assistant"):
            st.markdown("🧠 *Thinking…*")
    # call model
    reply = call_llm(user_text)
    # persist assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # clear the transient area (so we don't see double)
    thinking_slot.empty()

# ---------------- render the full history (single pass) ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
