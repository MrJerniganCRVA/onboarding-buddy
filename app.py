import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import load_chain

st.set_page_config(page_title="Onboarding Buddy", page_icon="🤝")
st.title("🤝 Onboarding Buddy")
st.caption("Your go-to for getting up to speed on the Payments team.")

# Sidebar persona selector
persona_label = st.sidebar.radio(
    "Persona",
    ["👥 Coworker", "📋 HR Partner"],
)
persona = "coworker" if "Coworker" in persona_label else "hr"

# Reload chain and clear history when persona changes
if st.session_state.get("persona") != persona:
    st.session_state.persona = persona
    with st.spinner(f"Switching to {persona_label} mode..."):
        st.session_state.chain = load_chain(persona)
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Active:** {persona_label}\n\n"
    + ("Answering as a senior teammate." if persona == "coworker"
       else "Answering as an HR Business Partner.")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the Payments team..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        stream = st.session_state.chain.stream({
            "question": prompt,
            "chat_history": chat_history,
        })
        answer = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": answer})
