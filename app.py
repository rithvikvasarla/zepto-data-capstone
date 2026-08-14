
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto Customer Support Assistant",
    version="1.0.0"
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str


class SupportResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: list[str]
    retrieved_ids: list[str]
    answer: str
    sources: list[str]
    confidence: float
    final_response: dict


# ============================================================
# MOCK LLM TOGGLE
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1")


# ============================================================
# EMBEDDING MODEL + CHROMADB
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="support_assistant/chroma_db"
)

collection = client.get_or_create_collection(
    name="zepto_policies"
)


# ============================================================
# POLICY KEYWORDS
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# ============================================================
# STRUCTURED PROMPT
# Required skeleton:
# Role - Context - Task - Format - Length
# Includes negative constraint + few-shot example
# ============================================================

STRUCTURED_PROMPT = """
ROLE:
You are Zepto's customer support assistant. Answer customer questions
accurately and only using the provided Zepto policy context.

CONTEXT:
The following text contains retrieved Zepto policy documents:

{context}

TASK:
Answer the customer's question using only the retrieved context.
If the answer is not present in the context, clearly state that the
provided policy documents do not contain the requested information.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided
context. Do not invent, assume, or hallucinate Zepto policies.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee for an order below INR 149?
Context: Standard delivery is free on orders over INR 149; orders below
this threshold incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.

FORMAT:
Return a concise answer in the required JSON structure with:
answer, sources, and confidence.

LENGTH:
Keep the answer concise and directly relevant to the customer's question.

CUSTOMER QUESTION:
{question}
"""


# ============================================================
# OPTIONAL REAL-LLM HELPER
# ============================================================

def call_real_llm(prompt: str) -> str:
    """
    Optional extension point for a real LLM when MOCK_LLM=0.

    The graded baseline never calls this function.
    """

    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set while MOCK_LLM=0."
            )

        client_llm = Groq(api_key=api_key)

        result = client_llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return result.choices[0].message.content

    except Exception as exc:
        raise RuntimeError(
            f"Real LLM call failed: {exc}"
        )


# ============================================================
# REAL LLM RESPONSE WITH VALIDATION + 2 RETRIES
# ============================================================

def generate_real_llm_response(
    prompt: str,
    source_ids: list[str]
) -> SupportResponse:

    last_error = None

    for attempt in range(3):

        try:

            if attempt == 0:
                current_prompt = prompt

            else:
                current_prompt = prompt + f"""

CORRECTIVE INSTRUCTION:
Your previous response did not match the required schema.

Return ONLY valid JSON with exactly these fields:
{{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}}

The confidence must be between 0 and 1.
Do not include Markdown fences.
"""

            raw = call_real_llm(current_prompt)

            import json

            data = json.loads(raw)

            response = SupportResponse(**data)

            return response

        except Exception as exc:
            last_error = exc

    return SupportResponse(
        answer=(
            "ERROR: The real LLM response could not be validated "
            "after 3 attempts."
        ),
        sources=source_ids,
        confidence=0.0
    )


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

def classify_intent(state: SupportState):

    query = state["query"].lower()

    # Required graded mock behavior
    if MOCK_LLM != "0":

        if any(
            keyword in query
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"
        else:
            intent = "general_question"

        return {
            "intent": intent
        }

    # Optional real-LLM extension
    prompt = f"""
Classify this Zepto customer query as exactly one of:

policy_question
general_question

Return only the classification.

Query:
{state["query"]}
"""

    raw = call_real_llm(prompt).strip().lower()

    if "policy_question" in raw:
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# NODE 2: RETRIEVE + ANSWER
# ============================================================

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    # --------------------------------------------------------
    # REAL RETRIEVAL
    # This always runs in both mock and real modes.
    # --------------------------------------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    ids = results["ids"][0]

    if not documents:

        response = SupportResponse(
            answer=(
                "The requested information was not found "
                "in the provided Zepto policies."
            ),
            sources=[],
            confidence=0.0
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "retrieved_documents": [],
            "retrieved_ids": [],
            "final_response": response.model_dump()
        }

    # --------------------------------------------------------
    # MOCK MODE - REQUIRED GRADED BASELINE
    # --------------------------------------------------------

    if MOCK_LLM != "0":

        top_chunk_snippet = documents[0][:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = SupportResponse(
            answer=answer,
            sources=ids,
            confidence=1.0
        )

    # --------------------------------------------------------
    # REAL LLM MODE - OPTIONAL
    # --------------------------------------------------------

    else:

        context = "\n\n".join(
            f"[{doc_id}]\n{doc}"
            for doc_id, doc in zip(ids, documents)
        )

        prompt = STRUCTURED_PROMPT.format(
            context=context,
            question=query
        )

        response = generate_real_llm_response(
            prompt,
            ids
        )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "retrieved_documents": documents,
        "retrieved_ids": ids,
        "final_response": response.model_dump()
    }


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(state: SupportState):

    # --------------------------------------------------------
    # MOCK MODE - REQUIRED GRADED BASELINE
    # --------------------------------------------------------

    if MOCK_LLM != "0":

        response = SupportResponse(
            answer=(
                "I can only answer questions about Zepto "
                "policies right now."
            ),
            sources=[],
            confidence=1.0
        )

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    else:

        prompt = f"""
ROLE:
You are Zepto's customer support assistant.

CONTEXT:
No policy retrieval was performed because this was classified
as a general question.

TASK:
Answer the user's question directly.

NEGATIVE CONSTRAINT:
Do not claim that information comes from Zepto policy documents.
Do not invent Zepto-specific policies.

FORMAT:
Return valid JSON with answer, sources, and confidence.

LENGTH:
Keep the response concise.

QUESTION:
{state["query"]}
"""

        response = generate_real_llm_response(
            prompt,
            []
        )

    return {
        "answer": response.answer,
        "sources": [],
        "confidence": response.confidence,
        "final_response": response.model_dump()
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_intent(
    state: SupportState
) -> Literal[
    "retrieve_and_answer",
    "direct_answer"
]:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(SupportState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.add_edge(
    START,
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

support_graph = builder.compile()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Zepto Customer Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask(request: AskRequest):

    result = support_graph.invoke({
        "query": request.query
    })

    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0)
    )
