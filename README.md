# AI Support Agent for AppleSupport (@AppleSupport)
**Hiver SDE Intern Take-Home Assignment**

An end-to-end, production-grade AI support agent pipeline for **AppleSupport** built on the Kaggle *Customer Support on Twitter* dataset (`twcs.csv`), featuring **few-shot intent classification**, **precedent-grounded RAG reply drafting**, and a **hybrid escalation engine with stated reasoning**.

---

## ⏱ Quickstart: Reproduce Headline Results in < 15 Minutes

### 1. Prerequisites & Setup
Ensure you have Python 3.9+ installed:
```bash
# Clone the repository
git clone <repo-url>
cd "AI Support agent"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (ultra-lightweight local ONNX embeddings & scikit-learn)
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy the template configuration:
```bash
cp .env.example .env
```
Open `.env` and configure your LLM provider:
```ini
# Recommended: Auto mode (probes local LM Studio first, falls back to OpenRouter)
# Routes LLM-as-Judge to OpenRouter for model independence from the drafting model.
LLM_PROVIDER=auto

# Local LM Studio (primary agent runtime on Apple Silicon, e.g. gemma-4-e4b)
# Requires LM Studio installed with google/gemma-4-e4b model downloaded.
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=google/gemma-4-e4b

# OpenRouter API (cloud fallback for agent + independent LLM-as-Judge)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```
*(Deterministic SQLite caching is enabled by default in `data/cache.sqlite`, so evaluation re-runs execute in $< 5$ seconds from cache.)*

---

### 3. Run Headline Evaluation Harness
To reproduce all headline metrics across Baseline A, Baseline B, System LLM (Gemma 4B via LM Studio / OpenRouter fallback), Escalation, and LLM-as-Judge (`nvidia/nemotron-3-super-120b-a12b:free`):
```bash
# Evaluates the full 162-item golden set (completes in <5 seconds from cache)
python -m src.eval.harness

# Or evaluate a rapid 36-query subsample for quick iteration:
# python -m src.eval.harness --limit 36
```

Expected headline benchmark summary (N=162):
```
--- CLASSIFICATION HEADLINE COMPARISON ---
Model                            | Accuracy   | Macro-F1
----------------------------------------------------------
Baseline A (Majority Class)      |    11.11% |     2.22%
Baseline B (TF-IDF + LogReg)     |    62.96% |    63.71%
System (Few-Shot LLM Agent)      |    56.17% |    55.85%

--- ESCALATION ENGINE PERFORMANCE ---
Accuracy:  51.85%
Precision: 21.21%
Recall:    100.00%
F1-Score:  35.00%

--- LLM-AS-JUDGE RUBRIC RATINGS (1-5 Scale) ---
  Groundedness:          4.77 / 5.0
  Correctness:           4.66 / 5.0
  Tone Match:            4.86 / 5.0
  Resolution Likelihood: 4.60 / 5.0
  Overall Mean Quality:  4.72 / 5.0

--- Human-vs-Judge Alignment (35 pairs) ---
  Pearson Correlation (r):          0.45
  Spearman Rank Correlation (rho):  0.26
  Quadratic Weighted Cohen's Kappa: 0.11
  Mean Absolute Error (MAE):        0.67
```

---

## 🏗️ Repository Architecture

```
/data/
  twcs_top_brands.json             # Kaggle 2.8M tweet distribution analysis
  candidate_brands_analysis.json   # Comparative metrics (Apple vs Spotify vs Amazon vs Uber)
  banking77/                       # Secondary dataset (train.csv, test.csv, categories.json)
  processed/
    apple_threads.jsonl            # 106,625 reconstructed conversation chains
    historical_kb.jsonl            # 80% chronological split (KB & baseline training)
    eval_pool.jsonl                # 20% held-out chronological slice (leakage-free)
    resolution_kb.json             # 1,500 resolved cases indexed by intent
    resolution_kb_vectors.npy      # Pre-computed 384-d dense BGE embeddings
/src/
  pipeline/
    reconstruct.py                 # Thread reconstruction & temporal partitioner
    kb_builder.py                  # Resolution KB builder using proxy heuristic
    agent.py                       # Unified AppleSupportAgent pipeline
  taxonomy/
    cluster.py                     # Unsupervised BGE embedding + K-Means silhouette sweep
    taxonomy_def.py                # 9-class empirical taxonomy definitions & exemplars
    curate_golden_set.py           # Stratified sampling from held-out slice
    verify_golden_cli.py           # Interactive CLI verification tool
  classify/
    baselines.py                   # Baseline A (Majority) & Baseline B (TF-IDF + LogReg)
    llm_classifier.py              # Few-shot LLM classifier with prompt caching
  draft/
    rag_drafter.py                 # Dense cosine retrieval (k=3) + grounded reply drafter
  escalate/
    rules.py                       # Deterministic regex rules (legal, theft, harassment)
    engine.py                      # Hybrid escalation engine with stated reasoning
  eval/
    metrics.py                     # Classification & escalation score calculators
    judge.py                       # LLM-as-Judge with independent model (4 dimensions)
    agreement.py                   # Pearson correlation & Cohen's kappa calculator
    harness.py                     # Master evaluation harness
  utils/
    embeddings.py                  # Local ONNX runtime BGE embedding generator
    llm_client.py                  # Unified client (LM Studio primary + OpenRouter fallback/judge + SQLite cache)
/golden_set/
  golden_eval_set.json             # 162 hand-curated & verified evaluation examples
  sampling_note.md                 # Documentation on sampling & labeling methodology
/report/
  report.md                        # Max 6-page comprehensive technical report
  decision_log.md                  # 13 non-obvious engineering decisions & trade-offs
README.md                          # Quickstart & reproduction guide
```

---

## 🎯 Deliverables Mapping

| Required Deliverable | Repository File / Artifact |
| :--- | :--- |
| **1. Runnable Pipeline (<15 min reproduction)** | `README.md` & `src/eval/harness.py` |
| **2. Golden Evaluation Set (150–250 examples)** | [`golden_set/golden_eval_set.json`](golden_set/golden_eval_set.json) & [`golden_set/sampling_note.md`](golden_set/sampling_note.md) |
| **3. Evaluation Harness & Evidence of Judge Agreement** | [`src/eval/harness.py`](src/eval/harness.py), [`src/eval/judge.py`](src/eval/judge.py), [`src/eval/agreement.py`](src/eval/agreement.py) |
| **4. Technical Report (max 6 pages)** | [`report/report.md`](report/report.md) |
| **5. Decision Log (13 non-obvious decisions & trade-offs)** | [`report/decision_log.md`](report/decision_log.md) |

---

## 🧪 Interactive Single Query Demo

You can interactively test the agent on any custom customer tweet:
```bash
python -c "
from src.pipeline.agent import AppleSupportAgent
agent = AppleSupportAgent()
query = 'My iPhone 8 battery drops 50% in 1 hour after updating to iOS 11'
print(agent.process_message(query))
"
```
