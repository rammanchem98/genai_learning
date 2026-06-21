# Generative AI & Agentic Systems Learning Roadmap

A hands-on, code-backed repository tracking my progress through production RAG architecture, function calling mechanics, and Agentic workflows.

---

##  My Daily Progress Checklist

###  Phase 1: Vector Storage & Production RAG
- [x] Set up BigQuery data extraction pipeline
- [x] Format unstructured document payloads for high-fidelity embedding
- [x] Implement batched processing using `gemini-embedding-2`
- [x] Handle Google SDK type constraints using nested `types.Content` and `types.Part`
- [x] Manage API server rate limits (`429 Resource Exhausted`) using backoff strategies
- [x] Set up localized semantic search with `chromadb` vector indexing
- [x] Connect vector search outputs directly to LLM context windows for absolute grounding

###  Phase 2: Function Calling & Manual ReAct Loops (Current)
- [x] Define isolated Python Tool functions with strict semantic docstrings
- [x] Disable Automatic Function Calling (`client.automatic_function_calling = False`)
- [x] Manually intercept and parse raw `FunctionCall` instruction packets from the LLM
- [x] Construct the local application loop to execute tools on behalf of the agent
- [x] Pipe manual `FunctionResponse` packets back to the LLM to finish the ReAct cycle

###  Phase 3: Memory, Chat History & State Management
- [x] Understand the difference between stateless calls and stateful conversations
- [x] Implement a system-managed rolling chat memory array
- [x] Handle multi-turn historical contexts inside an agent loop

###  Phase 4: Production MLOps & Architecture (Certification Guardrails)
- [ ] Deep dive into Google Cloud Professional ML Engineer exam core patterns
- [ ] Study pipeline automation, monitoring data drift, and deployment safety

---

## 🛠️ Local Project Workspace Index

- 📁 `chroma_db/` - Local Vector database storing 100 high-dimensional world air quality vector points.
- 📄 `ingest_vectors.py` - Core ETL script executing BigQuery queries, batch embedding generation, and Chroma DB updates.
- 📄 `query_vector.py` - The live testing playground for manual ReAct loops and function-calling logic.
- 📄 `inspect_db.py` - Administration utility file to quickly drop collections and debug vector shapes.