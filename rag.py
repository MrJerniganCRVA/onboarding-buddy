from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are Onboarding Buddy, a friendly and knowledgeable AI assistant \
for new employees joining the Payments team at Acme Corp. Your job is to help them \
get up to speed quickly by answering questions about the team, its people, processes, \
technology, and current work — using only the information provided to you.

Guidelines:
- Be warm, concise, and helpful.
- Ground every answer in the provided context. Do not invent facts.
- If you don't have enough information to answer confidently, say so clearly and \
suggest who the new employee might ask (e.g., a specific team member).
- When naming people or referencing ownership, be specific.
- Keep answers focused and scannable — use bullet points when listing multiple items."""


def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.3,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    SYSTEM_PROMPT + "\n\nContext from team documents:\n{context}"
                ),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        },
    )

    return chain
