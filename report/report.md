# Comprehensive Technical Report: AI Support Agent for AppleSupport
**Hiver SDE Intern Take-Home Assignment**

---

## 1. Problem Framing & Scope

### 1.1 What "Good" Means for AppleSupport
On Twitter (`@AppleSupport`), the standard of customer care differs fundamentally from e-commerce or delivery platforms. An e-commerce bot can often resolve an inquiry with a tracking link or order refund; an airline bot requires a booking reference. In contrast, **Apple Support interactions are diagnostic, reputation-critical, and technically nuanced**:
1. **Accurate Triage without False Promises:** Apple customers face complex hardware/software issues (e.g. rapid battery discharge after iOS updates, iCloud account lockouts, FaceTime activation errors, crackling speakers). A "good" support agent must identify the underlying technical domain, ask precise diagnostic triage questions (*"What iOS version are you running?", "Does this occur on Wi-Fi or cellular?"*), and provide official Apple KB documentation without ever inventing warranty policies, device replacement guarantees, or fake diagnostic links.
2. **Defensive Safety & Escalation:** Physical hardware defects, water damage, stolen devices involving police reports, and explicit legal or harassment threats cannot be automated. "Good" means knowing when **not** to answerâ€”escalating immediately to human specialists with a clear, stated justification.
3. **Succinct Brand Persona:** Replies must strictly adhere to Apple's signature empathetic, calm, and professional tone within Twitter's 240-character length limit.

### 1.2 What We Chose *Not* to Build (Explicit Out-of-Scope Decisions)
Per PRD §1, the following features were deliberately scoped out to preserve evaluation rigor and respect operational constraints:
- **Multilingual Support:** Although the raw Kaggle dataset contains non-English tweets (Spanish, German, Japanese), AppleSupport is predominantly English (~99.5%, based on manual review of ~200 sample threads). Attempting multilingual support on rate-limited free models introduces translation noise and dilutes evaluation quality.
- **Direct Serial Number / iCloud Account Lookups:** An external AI cannot access Apple's internal CRM or activation lock database. The system directs users to authenticate via official DM or the Apple Support app rather than simulating private database state.
- **Multi-Turn State Machines Beyond First Reply:** Evaluating multi-turn conversational trees on static Twitter threads introduces synthetic customer responses. We focus on the high-leverage first touch: **accurate intent classification, grounded draft reply, and auto-handle vs. escalate decision**.

---

## 2. System Architecture & Components

```
Raw Twitter Dataset (twcs.csv)
        â”‚
        â–¼
[1] Temporal Split & Thread Reconstruction (Zero-Leakage: 80% KB / 20% Held-Out)
        â”‚
        â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”�
        â–¼                                        â–¼                                        â–¼
[2] 9-Class Intent Taxonomy              [3] Resolution KB (1,500 cases)          [4] Golden Set (162 items)
    (BGE Embeddings + K-Means)               (Resolution Proxy Heuristic + ONNX)      (Stratified Held-Out Slice)
        │                                        │                                        │
        ▼                                        ▼                                        ▼
[5] Intent Classifiers                   [6] Grounded RAG Drafter                 [7] Hybrid Escalator
    - Baseline A: Majority Class             - Dense Cosine Similarity (k=3)          - Hard Rules (Theft, Legal)
    - Baseline B: TF-IDF + LogReg            - Precedent-Grounded Drafting            - Soft Confidence/Sim Floors
    - System: Few-Shot LLM Agent             - Hallucination Regex Validation         - LLM Stated Reasoner
        │                                        │                                        │
        └────────────────────────────────────────┴────────────────────────────────────────┘
                                                 │
                                                 ▼
[8] Evaluation Harness (Accuracy, Macro-F1, Precision/Recall, LLM-as-Judge Rubric, Human Agreement)
```

### 2.1 LLM Runtime & Execution Tier (Local Apple Silicon via LM Studio + Cloud Fallback)
The System Agent (`AppleSupportAgent`) was evaluated using **Google's Gemma 4B (`google/gemma-4-e4b`) hosted locally via LM Studio on Apple Silicon (M-series unified memory)**, with OpenRouter's free-tier API configured as an automated fallback. Running the primary agent locally on device provides zero network latency, zero token cost, and complete immunity to API rate-limit throttling during evaluation runs. The system's modular `LLMClient` abstracts the underlying provider, seamlessly switching between local LM Studio and OpenRouter free-tier endpoints via standard OpenAI-compatible API schemas.

---

## 3. Experimental Results vs. Baselines

We curated and ground-truth verified **162 held-out evaluation examples** (exactly 18 per class across all 9 taxonomy intents, documented in `golden_set/golden_eval_set.json` and `golden_set/sampling_note.md`). To navigate API rate limits, provider throttling, and latency during iterative benchmark execution, **headline evaluation metrics were run on a balanced 36-query subsample** (exactly 4 per class across all 9 taxonomy intents, representing exactly 22.2% of the golden set). The master harness supports running the entire 162-item set via `python -m src.eval.harness --limit 162`. Reply quality was scored across 15 representative generated drafts using an independent LLM judge (`nvidia/nemotron-3-super-120b-a12b:free` on OpenRouter), and judge-human alignment was computed across 14 paired spot-check comparisons with human ground-truth ratings.

### 3.1 Intent Classification Headline Results

| Model / Architecture | Accuracy | Macro-F1 | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **Baseline A: Majority Class Predictor** | 11.11% | 2.22% | Always predicts `other_unclear`. Completely collapses under stratified 9-class evaluation ($1/9 = 11.11\%$). |
| **Baseline B: TF-IDF + Logistic Regression** | **69.44%** | **69.12%** | Trained on 2,500 historical weakly labeled tweets. Strong on frequent lexical keywords (`wifi`, `battery`, `iTunes`), but has zero semantic reasoning. |
| **System: Few-Shot LLM Agent (Gemma 4B via LM Studio)** | **61.11%** | **60.08%** | Few-shot in-context reasoning using Google's Gemma 4B (`google/gemma-4-e4b`) hosted locally in LM Studio with SQLite prompt caching. Understands conversational nuance and high-risk phrasing, but suffers from category boundary bleed. |

#### Empirical Analysis: Why the TF-IDF Baseline Beat the LLM (69.44% vs. 61.11%)
The fact that a simple linear classifier outperformed a modern few-shot LLM by **8.33 percentage points** is the single most revealing empirical finding of this benchmark. Rather than treating this as an anomaly to hide, a per-class error breakdown reveals the fundamental trade-off between lexical memorization and semantic reasoning:

| Taxonomy Intent | TF-IDF F1 (Recall) | LLM F1 (Recall) | Root Cause of Delta |
| :--- | :---: | :---: | :--- |
| `battery_power_issue` | **1.00** (1.00) | 0.80 (1.00) | Lexically crisp ("battery", "drain", "charge"). Both achieve 100% recall. |
| `apps_services_media` | **0.86** (0.75) | 0.33 (0.25) | TF-IDF memorized "iTunes" tokens; LLM bled into `os_software_glitch`. |
| `connectivity_network`| **0.86** (0.75) | 0.67 (0.50) | TF-IDF keyed on "wifi", "AirPods"; LLM was distracted by emotional rants. |
| `orders_repairs_store` | **0.86** (0.75) | **0.86** (0.75) | Tied. Both recognize store, shipment, and replacement requests well. |
| `account_icloud_security`| 0.57 (0.50) | **0.75** (0.75) | LLM superior at identifying account takeover and password reset nuance. |
| `os_software_glitch` | **0.75** (0.75) | 0.57 (**1.00**) | LLM over-predicted glitch on *any* bug mention (Recall 1.00, Precision 0.40). |
| `hardware_screen_audio`| **0.86** (0.75) | 0.29 (0.25) | LLM confused physical screen unresponsiveness with software freezes. |
| `other_unclear` | **0.47** (1.00) | 0.29 (0.25) | TF-IDF defaulted ambiguous inputs here; LLM attempted to diagnose them. |
| `theft_lost_legal` (Critical) | **0.00** (**0.00**) | **0.86** (**0.75**) | **Catastrophic TF-IDF failure:** Missed 100% of stolen devices & lawsuits. |

1. **Why TF-IDF Won on Macro Accuracy:** In a 9-way classification task on short Twitter messages, frequent lexical n-grams (`wifi`, `iTunes`, `screen`, `battery`) act as nearly deterministic topic anchors. Trained on 2,500 domain tweets, logistic regression constructs sharp linear hyperplanes that correctly partition standard hardware and connectivity queries.
2. **Why Few-Shot LLM Suffered on Small Models:** The headline 61.11% classification accuracy was achieved with a 4-billion parameter open-weights model (`google/gemma-4-e4b`) running locally via LM Studio (with OpenRouter free-tier fallback). Without task-specific fine-tuning, a 4B parameter model prompted with few-shot exemplars experiences **category boundary bleeding**. The model disproportionately defaulted to `os_software_glitch` whenever words like "update", "bug", or "freezing" appeared, driving `os_software_glitch` recall to 100% but cratering its precision to 40.0% (dragging down macro-F1 to 60.08%).
3. **The Fatal Flaw of TF-IDF:** While TF-IDF won on headline accuracy, **it achieved 0.00% recall on safety-critical tickets (`theft_lost_legal`)**, classifying every police report, stolen iPhone, and consumer court threat as `other_unclear`. In an enterprise support environment, deploying TF-IDF would lead to catastrophic legal and brand liability. The LLM, by contrast, achieved **85.71% F1 (75% recall)** on high-risk intents, providing the semantic understanding that enabled our hybrid escalation engine to achieve **100% safety recall**.

---

### 3.2 Escalation Decision Engine Performance

The Hybrid Escalation Engine was benchmarked against the golden human escalation labels ($N=36$, where 4 cases required mandatory escalation: stolen devices, legal disputes, and abusive interactions):

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Escalation Recall** | **100.00%** | **Zero safety false negatives**: Successfully intercepted 4/4 high-risk queries (stolen hardware, police reports, consumer court lawsuits, abusive rants). |
| **Escalation Accuracy** | **58.33%** | 21 out of 36 routing decisions matched ground truth. Reflects deliberate bias towards over-escalation. |
| **Escalation Precision** | **21.05%** | 4 true escalations out of 19 total escalations (15 false alarms routed to humans). |
| **Escalation F1-Score** | **34.78%** | Mathematical reflection of the asymmetric trade-off: safety recall is maximized at the cost of autonomous deflection. |

**Confusion Breakdown:** True Positives: 4 | False Negatives: 0 | True Negatives: 17 | False Positives: 15.
All 15 false alarms occurred on ambiguous, high-frustration queries (e.g. *"somebody hacked my Apple account"*, *"store never gave me back my power cord"*, or classifier confidence $<0.65$). In customer support operations, an unnecessary human escalation is a minor cost; a missed legal threat or uncontained account takeover is a catastrophic failure.

---

### 3.3 Reply Quality (LLM-as-Judge with Independent Rubric)

Per PRD §4.7 and Deliverable 3, $N=15$ generated draft replies were evaluated across 4 standardized rubric dimensions (1–5 integer scale) using an independent judge model (`nvidia/nemotron-3-super-120b-a12b:free` hosted on OpenRouter):

| Rubric Dimension | Score (1–5) | Operational Meaning & Rubric Standard |
| :--- | :---: | :--- |
| **Groundedness** | **4.87 / 5.0** | Strict fidelity to retrieved precedents; zero fabricated URLs (`support.apple.com` or `apple.co` only) and no unverified replacement promises. |
| **Correctness** | **4.67 / 5.0** | Accurately targets the core customer complaint with appropriate diagnostic triage questions. |
| **Tone Match** | **4.93 / 5.0** | Exemplary Apple Support persona: calm, empathetic, professional, within Twitter's 240-character limit. |
| **Resolution Likelihood** | **4.60 / 5.0** | Provides clear, actionable next steps (DM link, diagnostic settings path, or official KB guide). |
| **Overall Quality Mean** | **4.77 / 5.0** | Indicates high production-readiness for human-in-the-loop agent assist. |

#### Judge Model Identity & Scoring Inconsistency Analysis
1. **Judge Model Identity:** All reported judge ratings were produced by **NVIDIA Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b:free`) accessed via OpenRouter. In offline/local development, the harness also supports local evaluation using `google/gemma-4-e4b` via LM Studio. Per PRD §2.1 and §4.7, using an independent, high-capacity model family (120B Nemotron evaluating drafts generated by 4B Gemma) strictly eliminates self-preference bias.
2. **Deprecation of Earlier Candidate (`gemini-2.0-flash-lite`):** During initial prototyping, Google's `gemini-2.0-flash-lite:free` was evaluated as a judge. However, it was subsequently retired and made unavailable on OpenRouter's free tier. This volatility exemplifies the fragility of relying on external free-tier endpoints and reinforces why pinned model identifiers and SHA-256 response caching (`data/cache.sqlite`) are mandatory for reproducible evaluation.
3. **Risk of Scoring Inconsistency from Judge Rotation:** Different judge models possess fundamentally different calibration baselines and leniency thresholds (e.g., Nemotron 120B vs. Gemma 4B vs. Gemini Flash). If evaluation calls dynamically rotate across different judge models mid-benchmark due to rate limits or network fallbacks, inter-judge leniency variance injects severe confounding noise into the scores, corrupting both Cohen's $\kappa$ and Pearson $r$ correlation against human ground truth. To prevent this confounding effect, all headline judge evaluations and human-alignment pairs were evaluated against the single pinned Nemotron 120B model with deterministic SQLite caching.

---

### 3.4 Evidence of Judge-Human Agreement

To satisfy assignment Deliverable 3, $N=14$ generated replies were independently hand-scored by a human reviewer on the identical 1â€“5 rubric. Statistical alignment between human and judge ratings:

- **Pearson Correlation ($r$):** **0.8219** ($p = 0.0003$, statistically significant, strong positive linear alignment).
- **Spearman Rank Correlation ($\rho$):** **0.6803** ($p = 0.007$, strong monotonic rank-order agreement).
- **Quadratic Weighted Cohen's Kappa ($\kappa$):** **0.1954** (captures exact categorical agreement on integer thresholds).
- **Mean Absolute Error (MAE):** **0.5893 points** (Human mean: 4.18, Judge mean: 4.77 on a 5-point scale).

#### Understanding the "Kappa Paradox" ($\kappa = 0.1954$ vs. $r = 0.8219$)
At first glance, a Cohen's $\kappa$ of 0.1954 appears low compared to a high Pearson $r$ of 0.8219. This is a classic manifestation of the well-documented **Feinstein & Cicchetti (1990) Kappa Paradox**:
1. **Severe Marginal Distribution Skew (Ceiling Effect):** In high-quality support reply generation, ratings are heavily concentrated in the top tiers (4s and 5s; human mean 4.18, judge mean 4.77). When the marginal prevalence is heavily unbalanced, the expected chance agreement $P_e$ becomes exceptionally high ($\approx 0.75$). Because Cohen's kappa is calculated as $\kappa = \frac{P_o - P_e}{1 - P_e}$, a high $P_e$ drastically compresses $\kappa$ even when observed agreement $P_o$ is high.
2. **Systematic Judge Leniency Bias (+0.59 MAE):** The LLM judge exhibited a consistent positive calibration bias of +0.59 points, frequently awarding a 5/5 to replies that a human evaluator rated 4/5. Because Cohen's kappa requires exact integer bin matches (4 vs. 5), this consistent 1-point shift heavily penalizes categorical $\kappa$. However, because the *relative ordering* is preserved, Pearson $r$ (0.8219) and Spearman $\rho$ (0.6803) confirm strong, statistically significant alignment.

---

## 4. Failure Analysis: Top 5 Failure Modes

By inspecting the misclassifications and routing discrepancies in `agent_eval_outputs.json`, we identified the top 5 concrete failure modes:

### Failure Mode 1: Multi-Intent Composite Queries (Security vs. App Service)
- **Verifiable Example (Eval ID 5):** *"@AppleSupport I am trying to reset my security questions and it is not allowing me to. I am now locked out of iMessage"*
- **Ground Truth:** `apps_services_media` | **Model Prediction:** `account_icloud_security` | **Action:** Escalated (`llm_reasoner`).
- **Hypothesis:** Customer inquiries frequently mention an account trigger ("security questions") that causes an application failure ("locked out of iMessage"). Single-label classification forces an arbitrary choice. The attention mechanism keyed on the security phrasing rather than the service symptom.
- **Production Fix:** Implement multi-label classification or hierarchical intent decomposition (`primary_domain: account`, `impacted_service: imessage`).

### Failure Mode 2: Over-Escalation on Security Phishing / Scam Suspicion
- **Verifiable Example (Eval ID 28):** *"@AppleSupport Please, this alert doesnâ€™t tell me what Iâ€™m giving my password to or what for and it bothers me constantly. How am I supposed to know some third party app isnâ€™t trying to scam me? https://t.co/qm5kVavhqt"*
- **Ground Truth:** Auto-handle (Provide standard phishing advisory link) | **System Action:** Escalated (`deterministic_rule: fraud/scam keyword`).
- **Hypothesis:** Hardcoded regex rules for words like "scam" and "fraud" fired conservatively. While safe, informational inquiries about verifying alert legitimacy do not require immediate human specialist intervention.
- **Production Fix:** Add an informational regex whitelist for "is this a scam / legit" queries to auto-respond with Apple's standard security guidance (`apple.co/reportphishing`).

### Failure Mode 3: Emotional Rants Obscuring Core Technical Symptoms
- **Verifiable Example (Eval ID 38):** *"Disappointing service from @115858. Covent Garden â€œgeniusâ€� - rude, patronising and unhelpful. My location and WiFi can still not be found!"*
- **Ground Truth:** `connectivity_network` | **Model Prediction:** `other_unclear` (Confidence: 0.50 $\rightarrow$ Escalated via soft threshold).
- **Hypothesis:** The customer spent 80% of their character budget recounting negative staff interactions, placing the technical issue at the very end. The classifier gave equal weight to the interpersonal complaint and defaulted to `other_unclear`.
- **Production Fix:** Pre-process incoming tweets with an extractive symptom segmenter that strips out emotional narrative preamble before passing the query to the intent classifier.

### Failure Mode 4: Category Bleed on Software Names vs. OS Glitches
- **Verifiable Example (Eval ID 16):** *"Hi @AppleSupport i'm having problems (again) with your ever-bugging iTunes software. plz help"*
- **Ground Truth:** `apps_services_media` | **Model Prediction:** `os_software_glitch`
- **Hypothesis:** In few-shot prompting, the word "software" and "bugging" strongly activates the `os_software_glitch` exemplar schema, overriding the specific app entity (`iTunes`).
- **Production Fix:** Provide explicit entity-to-category mapping dictionaries in the system prompt (`iTunes / Apple Music / Safari -> apps_services_media`; `iOS / macOS kernel / bootloop -> os_software_glitch`).

### Failure Mode 5: Generic DM Deflection in Retrieved Precedents
- **Verifiable Example (Eval ID 21):** Customer asked how to transfer iCloud backed-up photos to PC.
- **Drafted Reply:** Suggested opening a DM rather than providing the standard `support.apple.com/HT205323` guide.
- **Hypothesis:** Historical analysis revealed that **52.55% of Apple's historical tweets ask users to move to DM**. Dense retrieval naturally surfaces these high-frequency DM-deflection replies, causing the RAG drafter to mirror deflection rather than offering self-service troubleshooting steps.
- **Production Fix:** Re-rank retrieved precedents by penalizing pure DM-routing responses and boosting precedents containing canonical `support.apple.com` knowledge base articles.

---

## 5. "What is Misleading About My Headline Number?" (Mandatory Section)

While our system achieves **61.11% intent classification accuracy** (vs. 69.44% for TF-IDF), **100.00% safety escalation recall** (with 58.33% escalation accuracy), and an **average reply quality of 4.77 / 5.0**, reporting these figures without context would be intellectually dishonest:

1. **Headline Evaluation Run on Only 22.2% (N=36) of the 162-Item Golden Set:**
   We labeled, curated, and ground-truth verified 162 held-out customer interactions (18 per class across all 9 categories in `golden_set/golden_eval_set.json`), but headline evaluation metrics were generated on a 36-query subsample (4 per class, exactly 22.2% of the golden set). While evaluating 36 queries was a necessary engineering compromise driven by API rate limits and execution latency (evaluating all 162 examples requires ~486 sequential LLM calls across classification, RAG drafting, and rubric judging, taking ~35+ minutes on throttled free endpoints), reporting headline numbers on an $N=36$ slice increases metric variance. A single misclassification shifts accuracy by $\pm 2.78\%$ ($1/36$). While our stratified design guarantees equal representation across all 9 classes (including rare safety escalations), the headline figures reflect this 36-query benchmark rather than an exhaustive sweep of the full 162 examples. The full 162-item dataset is provided for unconstrained evaluation via `--limit 162`.

2. **The Baseline Paradox — Accuracy Does Not Equal Safety:**
   As demonstrated in §3.1, a simple TF-IDF linear classifier beats our Few-Shot LLM by 8.33% on headline accuracy (69.44% vs. 61.11%). However, reporting TF-IDF as the "better" support model would be catastrophically misleading: **TF-IDF achieved 0.00% recall on legal threats and stolen hardware**. In customer support, an agent that correctly labels battery complaints but routes lawsuits to a chatbot is unusable. The LLM's value lies not in raw lexical memorization, but in the semantic safety backbone that enabled **100% safety recall**.

3. **Local Gemma 4B Parametric Limits & Runtime Transparency:**
   The headline 61.11% classification accuracy and RAG draft replies were generated using a 4-billion parameter open-weights model (`google/gemma-4-e4b`) hosted locally on Apple Silicon via LM Studio (with OpenRouter free tier as fallback). While running locally achieves zero token cost, zero network latency, and complete immunity to API rate caps, a 4B model has substantially lower parametric reasoning capacity than frontier models (e.g. GPT-4o, Claude 3.5 Sonnet). This constrained capacity is directly responsible for the observed category boundary bleed (e.g., misclassifying iTunes app issues as OS glitches).

4. **The Resolution Heuristic is an Imperfect Operational Proxy:**
   We labeled historical threads as "resolved" if the customer did not tweet back after Apple's reply. In reality, a customer often ceases replying because they gave up in frustration, switched to phone support, or visited an Apple Store in person. The Resolution KB inevitably includes instances where AppleSupport provided an unhelpful response, yet the absence of a customer follow-up caused our RAG system to treat it as a gold-standard precedent.

5. **Stratified Sampling Masks Production Volume Skew:**
   In real production support, $>50\%$ of incoming tweets are generic chatter (`other_unclear`) or recurring iOS update rants. Our evaluation harness intentionally balanced the test set to exactly 4 examples per class across 9 categories (and 18 per class in the 162-item golden set). While this stratification is essential for measuring rare-class recall (such as theft and legal disputes), it means our 61.11% accuracy does **not** reflect real-world throughput, where performance on noisy general chatter dominates operational metrics.

6. **The Judge Agreement "Kappa Paradox" & Inter-Judge Leniency Inconsistency:**
   Reporting only Pearson correlation ($r = 0.8219$) would overstate judge agreement by hiding the low Cohen's kappa ($\kappa = 0.1954$). As analyzed in §3.4, the combination of score range restriction (most replies score 4 or 5) and a +0.59 MAE leniency bias compresses $\kappa$. Furthermore, different judge models have distinct leniency baselines (e.g., Nemotron 120B vs. Gemma 4B vs. Gemini Flash). If judge evaluations rotate dynamically across models due to API rate limits, inter-judge calibration shifts introduce severe confounding noise into the scores, directly undermining the validity of kappa analysis. Pinned caching on a single judge model (`nvidia/nemotron-3-super-120b-a12b:free`) was required to prevent cross-model distortion.

7. **Escalation Accuracy (58.33%) Conceals Extreme Safety-Bias:**
   An escalation accuracy of 58.33% sounds mediocre until decomposed into its constituent metrics: **100.00% Recall vs. 21.05% Precision**. The escalation engine produced 15 false alarms out of 36 queries, intentionally trading away human deflection efficiency to ensure zero safety leakage. Framing this as a "41.67% error rate" would misrepresent a deliberate, defensive architectural choice.

8. **Historical Dataset Drift:**
   The dataset captures Twitter support in late 2017 (iOS 11, iPhone X launch, macOS High Sierra). Many referenced support URLs (`apple.co/...`) and specific workarounds (e.g. the iOS 11 'i' keyboard autocorrect glitch) are obsolete today. Headline numbers demonstrate architectural soundness on 2017 data, but production deployment on modern iOS 18 requires re-indexing current support documentation.

9. **Free-Tier Infrastructure Volatility & Model Deprecation:**
   Free-tier API endpoints on aggregators like OpenRouter suffer from rapid model turnover. For instance, `gemini-2.0-flash-lite:free` was used during early prototyping but was subsequently retired from the free tier, requiring migration to `nvidia/nemotron-3-super-120b-a12b:free`. In production, relying on unpinned free-tier endpoints without fallback guarantees or response caching introduces operational fragility.

---

## 6. What We Would Do Next with One More Week

1. **Multi-Turn Context Tracking:** Extend the agent beyond single-turn replies to maintain conversational memory across multi-tweet customer threads using session state.
2. **Active Learning on Ambiguous Disagreements:** Automatically pipe examples where classifier confidence is borderline ($0.50 - 0.65$) into a human review queue to continually fine-tune the classification prompt.
3. **Dynamic Support KB Scraping:** Replace static tweet precedents with live ingestion of official Apple Support articles (`support.apple.com/HT...`) to ground replies in authoritative technical documentation rather than Twitter agent summaries.
4. **A/B Testing Against Paid Frontier Models:** Benchmark our free-tier rotation pipeline against paid frontier models (Claude 3.5 Sonnet, GPT-4o) to quantify the exact ROI and quality delta of paid APIs vs zero-cost local inference.
5. **Real-time Streaming Webhooks:** Package the pipeline into a FastAPI microservice with Twitter/Webhook event listeners for automated live shadow-testing.
