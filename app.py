import streamlit as st
from rag import load_chain

st.set_page_config(page_title="Onboarding Buddy", page_icon="🤝")
st.title("🤝 Onboarding Buddy")
st.caption("Your go-to for getting up to speed on the Payments team.")

if "chain" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the Payments team..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": prompt})
            answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
