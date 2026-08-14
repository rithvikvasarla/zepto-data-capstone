# Zepto Customer Support Assistant

## Module 3 — Support Assistant

A RAG-based customer support assistant for Zepto using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.


## Architecture

The system follows this pipeline:

Ingestion → Embedding → Retrieval → Generation

### 1. Ingestion

The eight Zepto policy documents are stored in the `docs/` directory. Each document is loaded as a document chunk and used as the source content for the retrieval system.

### 2. Embedding

The `all-MiniLM-L6-v2` model from Sentence Transformers generates embeddings locally. No external embedding API is required.

### 3. Retrieval

The embeddings are stored in the ChromaDB collection `zepto_policies`. The `retrieve_and_answer` LangGraph node embeds a policy query and retrieves the top 3 most similar chunks.

### 4. Generation

The retrieved context is used by the `retrieve_and_answer` node to generate the final policy response. General questions are handled by the `direct_answer` node.

The LangGraph flow is:

START → classify_intent → retrieve_and_answer → END

or

START → classify_intent → direct_answer → END


## LangGraph and Intent Routing

The application uses a LangGraph `StateGraph` with three nodes:

- `classify_intent` — classifies the query as `policy_question` or `general_question`.
- `retrieve_and_answer` — retrieves the top 3 relevant policy chunks from ChromaDB and generates the policy response.
- `direct_answer` — handles general questions without retrieval.

A conditional edge from `classify_intent` routes the query to `retrieve_and_answer` or `direct_answer`.

## MOCK_LLM Mode

The application defaults to `MOCK_LLM=1`. This is the required offline graded mode and does not make any LLM API calls.

In mock mode, intent classification uses a deterministic keyword heuristic. Policy keywords include `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, and `support hours`.

For policy questions, retrieval still runs using the local Sentence Transformer model and ChromaDB. The answer follows the required deterministic format:

`Based on the retrieved context: <top retrieved chunk>`

For general questions, the fixed mock response is:

`I can only answer questions about Zepto policies right now.`

When `MOCK_LLM=0` is explicitly set, the optional real-LLM path can be used for classification and answer generation. The retrieval step remains local and unchanged.


## Structured Output

The final API response is validated using the Pydantic `SupportResponse` model with three fields:

- `answer` — the generated answer as a string.
- `sources` — the retrieved document or chunk IDs. This is empty for general questions.
- `confidence` — a numeric value between 0 and 1.

In mock mode, the response is generated deterministically and validated directly using Pydantic.

## Structured Prompt

The optional real-LLM path uses a structured prompt following the Role–Context–Task–Format–Length skeleton.

The prompt also contains an explicit grounding constraint:

> Do not answer using information that is not present in the provided context.

A few-shot example is also included to demonstrate the expected question, context, and answer format.

The prompt instructs the model to provide a concise answer using only the retrieved Zepto policy context.


## FastAPI

The application is exposed through a FastAPI `POST /ask` endpoint.

Run the server locally with:

```bash
uvicorn support_assistant.app:app --host 127.0.0.1 --port 8000
```

### Example 1 — Policy Question

Request:

```json
{
  "query": "What is the delivery policy?"
}
```

This query contains the `delivery` keyword and is routed to `policy_question`, followed by ChromaDB retrieval.

### Example 2 — General Question

Request:

```json
{
  "query": "What is the capital of India?"
}
```

This query is routed to `general_question` and handled by the `direct_answer` node without retrieval.

Both examples were tested with the default `MOCK_LLM` setting.


## Docker

A Dockerfile is included in the repository for local containerization.

Build the image with:

```bash
docker build -t zepto-support-assistant .
```

Run the container with:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

The container runs the FastAPI application on port `7860` using Uvicorn.

Docker was not executed in Google Colab because the Colab runtime used for this project did not have Docker installed. The Dockerfile is provided and configured for local Docker build and execution.

## Technologies Used

- Python
- FastAPI
- LangGraph
- ChromaDB
- Sentence Transformers
- Pydantic
- Uvicorn
- Docker


## API Test Results

The following tests were performed locally with the default `MOCK_LLM` setting.

### Test 1 — Policy Question

Request:

```json
{
  "query": "What is the delivery policy?"
}
```

Raw response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01", "doc_02", "doc_05"],
  "confidence": 1.0
}
```

This query was classified as a `policy_question` and routed through ChromaDB retrieval.

### Test 2 — General Question

Request:

```json
{
  "query": "What is the capital of India?"
}
```

Raw response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This query was classified as a `general_question` and routed to `direct_answer` without retrieval.

## Running Locally

Start the FastAPI server with:

```bash
uvicorn support_assistant.app:app --host 127.0.0.1 --port 8000
```

The API is available at `POST /ask`.

