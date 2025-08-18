import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Pinned Buttons Chat", layout="wide")
st.markdown("<style>.block-container{padding-bottom:160px!important}</style>", unsafe_allow_html=True)

# -------- state --------
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": str}]
if "quick_action" not in st.session_state:
    st.session_state.quick_action = None
if "just_handled" not in st.session_state:
    st.session_state.just_handled = False

# -------- model call --------
def call_llm(prompt: str) -> str:
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

# -------- pinned bottom-right buttons (compact pill) --------
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

# -------- gather input (buttons > chat) --------
typed = st.chat_input("Type your question…")
user_text = st.session_state.quick_action or typed
st.session_state.quick_action = None

# -------- handle one turn with live bubbles (no duplicates) --------
if user_text:
    st.session_state.just_handled = True

    # 1) show the user's bubble immediately (right/blue)
    with st.chat_message("user"):
        st.markdown(user_text)

    # 2) show assistant placeholder and "Thinking…" (left/black)
    assistant_slot = st.chat_message("assistant").empty()
    assistant_slot.markdown("🧠 *Thinking…*")

    # 3) call LLM and replace the placeholder with the answer
    reply = call_llm(user_text)
    assistant_slot.markdown(reply)

    # 4) persist both to history (so they appear on the next rerun only)
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "content": reply})

# -------- render history (single pass) --------
# Skip in the same run we just handled input (prevents duplicate bubbles now).
if not st.session_state.just_handled:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# reset flag so the next rerun renders history as usual
st.session_state.just_handled = False
