# Hybrid RAG System with LangChain, LangGraph, and Ollama

A Retrieval-Augmented Generation system that routes each incoming question to either **semantic vector search** (unstructured documents) or **text-to-SQL** (structured data), rather than relying on vector search alone. Built with LangChain and LangGraph, running entirely on free, local models via Ollama — no paid API required.

## Why hybrid?

Most RAG tutorials only handle one kind of question: "what does this document say?" Real applications also get questions like "how much revenue came from region X?" — which vector search answers poorly, since the answer isn't written anywhere as prose, it has to be calculated from raw rows. This system detects which kind of question it's facing and routes it to the appropriate retrieval strategy automatically.

## Architecture

```
                        ┌─────────────┐
                        │    START     │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   classify   │  <- LLM decides: vector or sql?
                        └──┬───────┬──┘
                 vector    │       │    sql
                 ┌─────────▼─┐   ┌─▼─────────────┐
                 │  vector_   │   │  sql_retrieve  │  <- LLM writes SQL,
                 │  retrieve  │   │                │     runs it, returns
                 └─────┬──────┘   └───────┬────────┘     real result rows
                       │                  │
                       └────────┬─────────┘
                                │
                         ┌──────▼──────┐
                         │   generate   │  <- LLM writes final answer
                         └──────┬──────┘     from retrieved context
                                │
                          ┌─────▼─────┐
                          │    END     │
                          └───────────┘

Every request is also logged to `query_log` (question, route, context, answer, latency).
```

## Tech stack

| Component | Tool |
|---|---|
| Orchestration | LangGraph (`StateGraph` with conditional edges) |
| Chat/reasoning model | Ollama running `llama3.2` (local, free) |
| Embeddings | Ollama running `nomic-embed-text` (local, free) |
| Vector store | LangChain `InMemoryVectorStore` |
| Structured data + logging | SQLite (`orders` table + `query_log` table) |
| Evaluation | Custom LLM-as-judge scorer (faithfulness, answer relevancy, context precision) |
| Visualization | NetworkX + Matplotlib |

**Note on evaluation:** the project originally targeted the RAGAS library for scoring, but RAGAS's internal dependency on `langchain_community`'s Vertex AI integration caused persistent, unresolvable import conflicts in the Colab environment used for this project. Rather than fight the dependency chain, the same three metrics (faithfulness, answer relevancy, context precision) were reimplemented as direct LLM-judge prompts, giving equivalent scoring without the fragile dependency.

## Results

A 6-question benchmark was run through the pipeline and scored on faithfulness, answer relevancy, and context precision (0–1 scale, averaged, on a 0.0–1.0 scale per metric).

**Before fixing the router prompt:**

| Metric | Score |
|---|---|
| Faithfulness | 30% |
| Answer relevancy | 40% |
| Context precision | 48% |

**Root cause analysis:** inspecting the logged queries revealed the router (running on a small local model) was misclassifying several questions — e.g. "How long does standard shipping take?" was routed to the SQL path instead of vector search, since it superficially resembles a lookup question. This produced nonsensical SQL results and low-quality answers downstream.

**Fix:** the router prompt was rewritten with explicit few-shot examples distinguishing "written policy facts that happen to mention a number" (→ vector) from "aggregate calculations over many rows" (→ sql). Retrieval depth was also increased from k=3 to k=5.

**After the fix:**

| Metric | Score |
|---|---|
| Faithfulness | 60% |
| Answer relevancy | 70% |
| Context precision | 75% |

This is a 2x improvement in faithfulness from a single, targeted prompt change — identified through log inspection rather than guesswork.

## Known limitations

- `llama3.2` is a small (~2B-3B class) local model; it is used for routing, SQL generation, answer writing, *and* evaluation judging. A larger model (e.g. GPT-4-class) would likely improve all scores further, at the cost of requiring a paid API or more compute.
- The evaluation judge is the same model family used for generation, which can introduce some correlated bias compared to using an independent, stronger judge model.
- The benchmark set is intentionally small (6 questions) for demonstration purposes; a production evaluation would use 30+ questions per the original project scope.

## Possible extensions

- Swap in a larger Ollama model (`llama3.1:8b` or similar) with GPU acceleration for improved answer quality.
- Expand the benchmark set and knowledge base for more statistically reliable evaluation scores.
- Add an interactive chat loop instead of calling the pipeline programmatically per question.
- Migrate from SQLite to PostgreSQL for a more production-representative structured data layer.

## Project files

- `setup_sample_data.py` — generates the sample knowledge base and orders database
- `hybrid_rag_colab.ipynb` — the full, runnable pipeline (Colab notebook, Ollama-based)
- `query_log_export.csv` — logged history of every question processed by the pipeline
- `ragas_scores_export.csv` — evaluation scores per benchmark question
