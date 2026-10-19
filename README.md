# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked corpus, compared on a complex insurance contract:
the **HDFC ERGO *my: Optima Secure* Policy Wording** (53 pages; definitions, benefits, waiting
periods, exclusions, claim procedures).

| | Pipeline 1 – Baseline | Pipeline 2 – Re-rank |
|---|---|---|
| Retrieve | `all-MiniLM-L6-v2` bi-encoder, FAISS `IndexFlatIP` (cosine), top-10 | same top-10 |
| Re-rank | — | `cross-encoder/ms-marco-MiniLM-L-6-v2` on 10 (query, chunk) pairs |
| Context | top-3 by cosine | top-3 by cross-encoder score |
| Generate | Gemini `gemini-3.1-flash-lite`, temperature 0 | identical prompt |

## Contents

```
RAG_Baseline_vs_Reranking.ipynb   # the whole assignment: code, comments, outputs, plots
optima-secure-revision-pw.pdf     # source document
data/eval_queries.csv             # 114 hand-labelled questions with verbatim gold markers
results/                          # every table the notebook produces (CSV/JSON), incl. LLM answers
figures/                          # rank shift, retrieval metrics, score distributions, latency, scaling
```

## Notebook sections

1. Setup · 2. PDF loading (header/footer clipped) · 3. Token-aware chunking (200 tok, 40 overlap)
4. FAISS index · 5. Pipelines 1 & 2 (timed per stage) · 6. LLM prompt
7. **Stress tests** – keyword trap, exception, negation, buried-in-list, plus 2 limitation cases; rank-shift table & chart
8. **Answer 1 vs Answer 2** with blind LLM-judge grades (`gemini-3.8-flash`) + manual review
9. **Aggregate evaluation** on 114 queries – Hit@1, Hit@3, MRR@10, Recall@10, wins/losses
10. Score distributions (why the cross-encoder separates relevant from irrelevant)
11. **Latency profiling** per stage
12. **Scaling** – cross-encoder cost vs chunks scored, re-rank depth ablation, FAISS cost vs corpus size, extrapolation to 10M chunks
13. Limitations · 14. Summary

## Run it

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env          # free key: https://aistudio.google.com/apikey
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

Runs on CPU. LLM responses are cached in `results/llm_cache.json` (delete it to regenerate);
calls are spaced to stay inside the Gemini free-tier limit of 5 requests/minute.
