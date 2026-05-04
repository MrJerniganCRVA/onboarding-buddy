from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from rag import load_chain

app = FastAPI(
    title="Onboarding Buddy API",
    description=(
        "RAG-powered onboarding assistant for the Payments team at Acme Corp. "
        "Retrieves context from internal team documents and generates grounded responses. "
        "Supports persona switching between a Coworker and HR Business Partner mode."
    ),
    version="1.0.0",
)

# Load both chains once at startup
_chains = {
    "coworker": load_chain("coworker"),
    "hr": load_chain("hr"),
}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[Message] = []


class ChatResponse(BaseModel):
    answer: str
    persona: str


def _build_history(chat_history: list[Message]):
    history = []
    for msg in chat_history:
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))
    return history


@app.post("/chat", response_model=ChatResponse, summary="Chat using the default Coworker persona")
def chat(request: ChatRequest):
    """
    Ask Onboarding Buddy a question using the **Coworker** persona.
    Responds like a friendly senior teammate with practical, grounded answers.
    """
    chain = _chains["coworker"]
    answer = chain.invoke({
        "question": request.question,
        "chat_history": _build_history(request.chat_history),
    })
    return ChatResponse(answer=answer, persona="coworker")


@app.post("/chat/{persona}", response_model=ChatResponse, summary="Chat using a specific persona")
def chat_with_persona(
    persona: Literal["coworker", "hr"],
    request: ChatRequest,
):
    """
    Ask Onboarding Buddy a question using a specific persona:

    - **coworker** — Friendly senior teammate. Practical, uses "we" and "our team."
    - **hr** — HR Business Partner. Professional, policy-aware, references proper channels.
    """
    if persona not in _chains:
        raise HTTPException(status_code=400, detail=f"Unknown persona: {persona}")
    chain = _chains[persona]
    answer = chain.invoke({
        "question": request.question,
        "chat_history": _build_history(request.chat_history),
    })
    return ChatResponse(answer=answer, persona=persona)
