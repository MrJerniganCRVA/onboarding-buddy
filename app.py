import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag import load_chain

st.set_page_config(page_title="Payments Team Wiki", page_icon="📘")
st.title("Payments Team — Internal Wiki")
st.caption("Ask questions about team processes, policies, and how we work. Maintained by the Payments EM.")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] > section:first-child > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #D6DDE6;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}
[data-testid="stChatInput"] {
    border-top: 1px solid #D6DDE6;
    padding-top: 0.5rem;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.875rem;
    color: #4A5568;
}
</style>
""", unsafe_allow_html=True)

# Sidebar view selector
persona_label = st.sidebar.radio(
    "Answer style",
    ["Teammate", "HR"],
)
persona = "coworker" if persona_label == "Teammate" else "hr"

# Reload chain and clear history when view changes
if st.session_state.get("persona") != persona:
    st.session_state.persona = persona
    with st.spinner(f"Loading {persona_label} context..."):
        st.session_state.chain = load_chain(persona)
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**View:** {persona_label}\n\n"
    + ("Responses draw from team processes and general engineering context."
       if persona == "coworker"
       else "Responses draw from HR policies and people operations docs.")
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Search team docs, processes, and policies..."):
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
