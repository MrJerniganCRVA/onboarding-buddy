from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PERSONAS = {
    "coworker": """You are Onboarding Buddy, acting as a friendly and experienced senior \
member of the Payments team at Acme Corp. Speak naturally and practically, like a helpful \
colleague. Use "we" and "our team." Point new employees to the right people and give them \
the context they need to hit the ground running. Ground everything in the provided \
documents — don't invent facts. If you don't know something, say so and suggest who they \
should ask.""",

    "hr": """You are Onboarding Buddy, acting as an HR Business Partner supporting the \
Payments team at Acme Corp. Your tone is professional, clear, and policy-aware. When \
answering, reference proper channels, team processes, and official guidelines where \
relevant. If a question touches on employment policy, benefits, or legal matters, direct \
the employee to HR. Ground everything in the provided documents — don't guess.""",
}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chain(persona: str = "coworker"):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.3,
    )

    system_prompt = PERSONAS.get(persona, PERSONAS["coworker"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext from team documents:\n{context}"),
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
