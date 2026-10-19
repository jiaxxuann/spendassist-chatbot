# Description
Guides the development of the backend architecture and agent orchestrator.

# Instructions
When generating Python code for this project, adhere strictly to these rules:
- **Frameworks:** Use FastAPI for all REST endpoints and LangGraph for the tool-calling agent orchestrator.
- **Validation:** Use Pydantic models for all API request and response validation, and for all tool inputs/outputs.
- **Tools:** 
  - `query_spend`: A read-only tool for pre-approved, role-scoped database queries.
  - `draft_expense_claim`: A write-operation tool that strictly creates a draft awaiting manager approval, without executing a final submission.
- **Agent Logic:** Enforce a strict maximum step limit (recursion limit) in LangGraph to prevent infinite loops. Include logic to handle uncertain queries gracefully.
- **Retrieval (RAG):** Integrate ChromaDB as the local vector store. Use the `embeddinggemma-300m` embedding model. Apply structural chunking to policy documents, ensuring exact clause numbers are preserved as metadata and explicitly appended to final plain-language citations.