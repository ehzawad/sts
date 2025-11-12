# Idea Pipeline (dual STS + tag classifier)

Experimental question-answering pipeline that fuses a global semantic search leg (FAISS over multilingual-E5 embeddings) with a 195-tag classifier and reranks results with `DualSignalRanker`. Everything lives under `idea/` so the prototype stays self-contained.

## Prerequisites
- Python 3.10+, `pip`, and the `idea/datasets/` layout (`sts_train.csv`, `sts_eval.csv`, `tag_to_answer.json`, `features/*.json`) plus trained artifacts under `idea/models/semantic` and `idea/models/tag_classifier`.
- Create/activate a virtualenv and install the minimal runtime stack:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -U sentence-transformers torch faiss-cpu pandas numpy scikit-learn tqdm
  ```

## Core commands
- Generate / refresh n-gram features (writes to `idea/datasets/features/`):
  ```bash
  python idea/featurizer/generate_features.py --top-k 20 --auto-clean --dominance-ratio 2.0
  ```
- Train the 195-tag classifier and drop weights in `idea/models/tag_classifier/`:
  ```bash
  python idea/training/train_tag_classifier.py --models idea/models/tag_classifier --embedding-model intfloat/multilingual-e5-large-instruct --epochs 50 --batch-size 64 --lr 3e-4 --patience 12 --ngram-mode manual
  ```
- Build semantic FAISS indices and mapping files under `idea/models/semantic/`:
  ```bash
  python idea/training/build_faiss_indices.py --global
  ```
- Try the pipeline interactively (loads models + datasets from the paths above):
  ```bash
  python idea/examples/interactive_cli.py
  ```
- Run the pipeline programmatically:
  ```bash
  python - <<'PY'
  from idea import IdeaPipeline
  pipe = IdeaPipeline()
  pipe.initialize()
  response = pipe.run("How do I reset my MFA device?", fusion_top_k=3)
  print(response)
  PY
  ```

Tip: set `STS_EMBEDDING_DEVICE` to `cuda`, `mps`, `cpu`, or `auto` to control where the shared embedding model loads.
