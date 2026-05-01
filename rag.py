from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(
        model="llama3.2",
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nContext from team documents:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
