# RAG: baseline retrieval vs. cross-encoder re-ranking

Two retrieval pipelines over the same chunked document, compared on questions where plain dense retrieval fails.

* **Document:** *Twilight Imperium 4th Edition, Prophecy of Kings: Living Rules Reference 2.0* (44-page rulebook).
  The PDF is copyrighted, so it is not in the repo. The notebook downloads it from Fantasy Flight Games on the first run
  ([link](https://images-cdn.fantasyflightgames.com/filer_public/51/55/51552c7f-c05c-445b-84bf-4b073456d008/ti10_pok_living_rules_reference_20_web.pdf)).
* **Pipeline 1 (baseline):** `all-MiniLM-L6-v2` embeddings → FAISS inner product on normalised vectors (cosine) → top 10 → top 3 to the LLM.
* **Pipeline 2 (re-ranking):** the same top 10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs → top 3 to the LLM.
* **LLM:** `qwen2.5:7b` through Ollama, temperature 0, fixed seed, identical prompt for both pipelines.

Everything is in one notebook: [`rag_rerank.ipynb`](rag_rerank.ipynb). It is committed with its outputs (rank-shift tables, answers, latency).
The `results/` folder holds the same results as CSV.

## Run it

```bash
python3.12 -m pip install -r requirements.txt
ollama pull qwen2.5:7b          # needs the Ollama server running
jupyter nbconvert --to notebook --execute --inplace rag_rerank.ipynb   # or open it and Run All
```

A GPU is used if available (the full notebook takes a few minutes on an RTX 3070 Ti, 8 GB).
