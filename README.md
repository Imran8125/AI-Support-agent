# AI Support Agent for AppleSupport (@AppleSupport)
**Hiver SDE Intern Take-Home Assignment**

An end-to-end, production-grade AI support agent pipeline for **AppleSupport** built on the Kaggle *Customer Support on Twitter* dataset (`twcs.csv`), featuring **few-shot intent classification**, **precedent-grounded RAG reply drafting**, and a **hybrid escalation engine with stated reasoning**.

---

## Quickstart: Reproduce Headline Results in < 15 Minutes

### 1. Prerequisites & Setup
Ensure you have Python 3.9+ installed:
```bash
# Clone the repository
git clone https://github.com/Imran8125/AI-Support-agent.git
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

## Repository Architecture

```text
ai-support-agent/
├── data/
│   ├── banking77/                     # Secondary benchmark dataset (Banking77)
│   ├── eval_results/                  # Cached evaluation runs & benchmark reports
│   │   ├── agent_eval_outputs.json    # System agent test predictions on golden set
│   │   ├── baselines_results.json     # Baseline A & B evaluation outputs
│   │   └── benchmark_report.json      # Aggregated headline metric report
│   ├── processed/                     # Leakage-free processed artifacts & knowledge bases
│   │   ├── apple_threads.jsonl        # 106,625 reconstructed conversation chains
│   │   ├── historical_kb.jsonl        # 80% chronological split (KB & baseline training)
│   │   ├── eval_pool.jsonl            # 20% held-out chronological slice (evaluation pool)
│   │   ├── resolution_kb.json         # 1,500 resolved cases indexed by intent
│   │   └── resolution_kb_vectors.npy  # Pre-computed 384-d dense BGE embeddings
│   ├── candidate_brands_analysis.json # Empirical comparison of 4 candidate brands
│   ├── clustering_diagnostics.json    # K-Means silhouette sweeps across cluster counts
│   ├── taxonomy.json                  # Derived 9-class intent schema with descriptions
│   ├── twcs_top_brands.json           # Brand volume distribution analysis (2.8M tweets)
│   └── cache.sqlite                   # Deterministic LLM response cache
├── golden_set/
│   ├── golden_eval_set.json           # 162 hand-curated & verified evaluation examples
│   └── sampling_note.md               # Stratified sampling & annotation methodology
├── models/
│   └── bge-small/                     # Local ONNX runtime BAAI/bge-small-en-v1.5 model
├── report/
│   ├── decision_log.md                # 13 non-obvious engineering decisions & trade-offs
│   └── report.md                      # Comprehensive technical report (6-page limit)
├── scripts/                           # Exploratory data analysis & brand selection
│   ├── analyze_brand_candidates.py    # Multi-criteria scoring across top candidate brands
│   ├── analyze_apple_support.py       # Deep-dive analysis of AppleSupport conversational data
│   ├── analyze_banking77.py           # Exploration of Banking77 intent taxonomy
│   ├── analyze_kaggle_dataset.py      # Preliminary inspection of twcs.csv
│   └── analyze_twcs_deep.py           # Volume, thread-depth, and response time metrics
├── src/                               # Core agent source code
│   ├── classify/                      # Intent classification models
│   │   ├── baselines.py               # Baseline A (Majority) & Baseline B (TF-IDF + LogReg)
│   │   └── llm_classifier.py          # Few-shot LLM classifier with prompt caching
│   ├── draft/                         # Precedent-grounded reply generation
│   │   └── rag_drafter.py             # Cosine similarity retrieval (k=3) + grounded drafter
│   ├── escalate/                      # Human handoff & safety triggers
│   │   ├── rules.py                   # Deterministic regex patterns (legal, abuse, account takeover)
│   │   └── engine.py                  # Hybrid escalation engine with structured rationale
│   ├── eval/                          # Evaluation harness & rubric assessment
│   │   ├── agreement.py               # Human-vs-Judge correlation & Cohen's kappa metrics
│   │   ├── harness.py                 # Master reproducible evaluation runner
│   │   ├── judge.py                   # LLM-as-Judge rubric evaluator (4 dimensions)
│   │   └── metrics.py                 # Classification, escalation, and tone scoring
│   ├── pipeline/                      # Orchestration & data preparation
│   │   ├── agent.py                   # Unified AppleSupportAgent end-to-end interface
│   │   ├── kb_builder.py              # Resolution KB extraction using proxy heuristics
│   │   └── reconstruct.py             # Temporal thread reconstruction & 80/20 partitioner
│   ├── taxonomy/                      # Intent discovery & dataset curation
│   │   ├── cluster.py                 # Unsupervised BGE embedding + K-Means silhouette sweeps
│   │   ├── curate_golden_set.py       # Stratified sampling from held-out evaluation pool
│   │   ├── taxonomy_def.py            # 9-class empirical taxonomy definitions & exemplars
│   │   └── verify_golden_cli.py       # Interactive CLI tool for manual review & relabeling
│   └── utils/                         # Shared utilities & model clients
│       ├── embeddings.py              # Zero-dependency ONNX runtime embedding generator
│       └── llm_client.py              # Multi-provider client (LM Studio + OpenRouter + Cache)
├── .env.example                       # Environment variable configuration template
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation & quickstart guide
```

### Module Responsibilities

| Subsystem | Primary Path | Description & Role |
| :--- | :--- | :--- |
| **Data & Splits** | [`data/processed/`](data/processed/) | Leakage-free chronological partitions (80% historical KB / 20% held-out test), 1,500-entry resolution KB, and pre-indexed 384-d dense embeddings. |
| **Golden Evaluation Set** | [`golden_set/`](golden_set/) | 162 human-verified test conversations stratified across all 9 taxonomy classes with ground-truth intent, escalation tags, and human quality scores. |
| **Local Models & Cache** | [`models/`](models/), `data/cache.sqlite` | Offline ONNX runtime for BGE-small embeddings and persistent SQLite cache ensuring fast, deterministic LLM evaluation. |
| **Exploratory Scripts** | [`scripts/`](scripts/) | Quantitative analysis tools for brand selection, Twitter CS volume metrics, and candidate evaluation. |
| **Classification Engine** | [`src/classify/`](src/classify/) | 9-class intent classification comparing Majority (Baseline A), TF-IDF + Logistic Regression (Baseline B), and Few-Shot LLM prompting. |
| **Grounding & RAG Drafter** | [`src/draft/`](src/draft/) | Cosine-similarity retrieval over precedent resolutions ($k=3$) to synthesize grounded, brand-aligned AppleSupport responses. |
| **Escalation Engine** | [`src/escalate/`](src/escalate/) | Two-tier escalation engine combining fast deterministic safety rules with LLM sentiment & complexity analysis and explicit reasoning. |
| **Evaluation Harness** | [`src/eval/`](src/eval/) | End-to-end benchmark harness, multi-criteria LLM-as-Judge rubric, and statistical human-judge correlation validation. |
| **Pipeline & Orchestration**| [`src/pipeline/`](src/pipeline/) | End-to-end `AppleSupportAgent` facade, conversation thread reconstruction, and resolution KB compilation. |
| **Taxonomy & Curation** | [`src/taxonomy/`](src/taxonomy/) | Unsupervised K-Means clustering, silhouette optimization, taxonomy definition, and interactive curation CLI. |
| **Technical Reports** | [`report/`](report/) | Executive technical report (`report.md`) and comprehensive decision log (`decision_log.md`) documenting 13 architectural trade-offs. |

---

## Deliverables Mapping

| Required Deliverable | Repository File / Artifact |
| :--- | :--- |
| **1. Runnable Pipeline (<15 min reproduction)** | `README.md` & `src/eval/harness.py` |
| **2. Golden Evaluation Set (150–250 examples)** | [`golden_set/golden_eval_set.json`](golden_set/golden_eval_set.json) & [`golden_set/sampling_note.md`](golden_set/sampling_note.md) |
| **3. Evaluation Harness & Evidence of Judge Agreement** | [`src/eval/harness.py`](src/eval/harness.py), [`src/eval/judge.py`](src/eval/judge.py), [`src/eval/agreement.py`](src/eval/agreement.py) |
| **4. Technical Report (max 6 pages)** | [`report/report.md`](report/report.md) |
| **5. Decision Log (13 non-obvious decisions & trade-offs)** | [`report/decision_log.md`](report/decision_log.md) |

---

## Interactive Single Query Demo

You can interactively test the agent on any custom customer tweet:
```bash
python -c "
from src.pipeline.agent import AppleSupportAgent
agent = AppleSupportAgent()
query = 'My iPhone 8 battery drops 50% in 1 hour after updating to iOS 11'
print(agent.process_message(query))
"
```
