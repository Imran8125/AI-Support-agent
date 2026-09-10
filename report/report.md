# Comprehensive Technical Report: AI Support Agent for AppleSupport
**Hiver SDE Intern Take-Home Assignment**

---

## 1. Problem Framing & Scope

### 1.1 What "Good" Means for AppleSupport
On Twitter (`@AppleSupport`), the standard of customer care differs fundamentally from e-commerce or delivery platforms. An e-commerce bot can often resolve an inquiry with a tracking link or order refund; an airline bot requires a booking reference. In contrast, **Apple Support interactions are diagnostic, reputation-critical, and technically nuanced**:
1. **Accurate Triage without False Promises:** Apple customers face complex hardware/software issues (e.g. rapid battery discharge after iOS updates, iCloud account lockouts, FaceTime activation errors, crackling speakers). A "good" support agent must identify the underlying technical domain, ask precise diagnostic triage questions (*"What iOS version are you running?", "Does this occur on Wi-Fi or cellular?"*), and provide official Apple KB documentation without ever inventing warranty policies, device replacement guarantees, or fake diagnostic links.
2. **Defensive Safety & Escalation:** Physical hardware defects, water damage, stolen devices involving police reports, and explicit legal or harassment threats cannot be automated. "Good" means knowing when **not** to answer—escalating immediately to human specialists with a clear, stated justification.
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
        │
        ▼
[1] Temporal Split & Thread Reconstruction (Zero-Leakage: 80% KB / 20% Held-Out)
        │
        ├────────────────────────────────────────┬────────────────────────────────────────┐
        ▼                                        ▼                                        ▼
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

### 2.1 LLM Runtime & Execution Tier (Local Apple Silicon via LM Studio Primary + Cloud Fallback & Judge)
The System Agent (`AppleSupportAgent`) was evaluated using **Google's Gemma 4B (`google/gemma-4-e4b`) hosted locally via LM Studio on Apple Silicon (M-series unified memory)** as the primary runtime for agent generation (intent classification, RAG drafting, and escalation reasoning). Running high-volume agent tasks locally provides zero network latency, zero token cost, and complete immunity to API rate-limit throttling during iterative development. Meanwhile, OpenRouter's free-tier API is strategically reserved for the independent **LLM-as-Judge (`nvidia/nemotron-3-super-120b-a12b:free`)**, where independence from the drafting model (to prevent self-preference bias, PRD §2.1 & §4.7) matters far more than call volume, and serves as an automated cloud fallback if LM Studio is unreachable. The system's modular `LLMClient` abstracts the underlying providers, seamlessly coordinating local LM Studio and OpenRouter via `LLM_PROVIDER=auto` (with explicit `lmstudio` or `openrouter` overrides) in `.env`, backed by deterministic SHA-256 SQLite caching (`data/cache.sqlite`).

---

## 3. Experimental Results vs. Baselines

We curated and ground-truth verified **162 held-out evaluation examples** (exactly 18 per class across all 9 taxonomy intents, documented in `golden_set/golden_eval_set.json` and `golden_set/sampling_note.md`). To eliminate statistical sampling variance from earlier small-slice prototyping, **all headline evaluation metrics reported below were executed on the complete 162-item golden evaluation set** (18 verified examples per class across all 9 taxonomy intents, representing 100% of the curated golden set). Reply quality was evaluated across $N=35$ representative generated drafts using an independent LLM judge (`nvidia/nemotron-3-super-120b-a12b:free` on OpenRouter with local verification), and judge-human alignment was computed across all $N=35$ paired comparisons with human ground-truth hand-scored ratings, directly satisfying the PRD's 30–50 example target.

### 3.1 Intent Classification Headline Results (N = 162)

| Model / Architecture | Accuracy | Macro-F1 | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **Baseline A: Majority Class Predictor** | 11.11% | 2.22% | Always predicts `other_unclear`. Completely collapses under stratified 9-class evaluation ($1/9 = 11.11\%$). |
| **Baseline B: TF-IDF + Logistic Regression** | **62.96%** | **63.71%** | Trained on 2,500 historical weakly labeled tweets. Strong on frequent lexical keywords (`wifi`, `battery`, `iTunes`), but has zero semantic reasoning. |
| **System: Few-Shot LLM Agent (Gemma 4B via LM Studio)** | **56.17%** | **55.85%** | Few-shot in-context reasoning using Google's Gemma 4B (`google/gemma-4-e4b`) hosted locally in LM Studio with SQLite prompt caching. Understands conversational nuance and high-risk phrasing, but suffers from category boundary bleed. |

#### Empirical Analysis: Why the TF-IDF Baseline Beat the LLM (62.96% vs. 56.17%)
Evaluating across the complete 162-item dataset confirms that a simple linear classifier outperformed the 4B few-shot LLM by **6.79 percentage points** in headline accuracy (62.96% vs. 56.17%). Stating raw counts alongside percentages reveals the exact trade-off between lexical memorization and semantic reasoning across all 9 classes (Support $N=18$ per class):

| Taxonomy Intent | TF-IDF Correct (Recall) [F1] | LLM Correct (Recall) [F1] | Support | Root Cause of Delta |
| :--- | :---: | :---: | :---: | :--- |
| `account_icloud_security`| **15/18** (83.3%) [**0.86**] | 12/18 (66.7%) [0.71] | 18 | TF-IDF memorized "iCloud" / "Apple ID" tokens; LLM identified subtle lockout nuance. |
| `apps_services_media` | **11/18** (61.1%) [**0.69**] | 7/18 (38.9%) [0.45] | 18 | TF-IDF keyed on "iTunes" tokens; LLM bled into `os_software_glitch`. |
| `battery_power_issue` | 12/18 (66.7%) [**0.80**] | **16/18** (**88.9%**) [0.65] | 18 | LLM achieved 88.9% recall on battery queries; false alarms lowered F1. |
| `connectivity_network`| **10/18** (**55.6%**) [**0.65**] | 8/18 (44.4%) [0.62] | 18 | Close contest; TF-IDF anchored on "wifi", LLM achieved 100% precision (8/8). |
| `hardware_screen_audio`| **13/18** (**72.2%**) [**0.79**] | 4/18 (22.2%) [0.33] | 18 | LLM confused physical screen unresponsiveness with software freezes. |
| `orders_repairs_store` | 11/18 (61.1%) [**0.76**] | **13/18** (**72.2%**) [0.72] | 18 | LLM superior on store reservations and order shipment tracking. |
| `os_software_glitch` | 13/18 (72.2%) [**0.79**] | **15/18** (**83.3%**) [0.51] | 18 | LLM over-predicted glitch on general bug/freeze complaints (Recall 83.3%, Precision 36.6%). |
| `other_unclear` | **17/18** (**94.4%**) [**0.41**] | 5/18 (27.8%) [0.28] | 18 | TF-IDF defaulted ambiguous inputs here; LLM attempted to diagnose them. |
| `theft_lost_legal` (Critical) | **0/18** (**0.0%**) [**0.00**] | **11/18** (**61.1%**) [**0.76**] | 18 | **Catastrophic TF-IDF failure:** TF-IDF missed 100% of stolen devices & lawsuits. LLM achieved 100% precision (11 TP, 0 FP, 7 FN). |

1. **Why TF-IDF Won on Macro Accuracy:** On short customer tweets, frequent lexical n-grams (`wifi`, `iTunes`, `screen`, `battery`) act as deterministic topic anchors. Trained on 2,500 historical tweets, logistic regression constructs sharp linear hyperplanes that partition standard hardware and connectivity queries effectively.
2. **Why Few-Shot LLM Suffered on Small Models:** The headline 56.17% classification accuracy was achieved with a 4-billion parameter open-weights model (`google/gemma-4-e4b`). Without task-specific fine-tuning, a 4B parameter model prompted with few-shot exemplars experiences **category boundary bleeding**. The model disproportionately defaulted to `os_software_glitch` whenever words like "update", "bug", or "freezing" appeared, driving `os_software_glitch` recall to 83.3% (15/18) but cratering its precision to 36.59% (dragging down macro-F1 to 55.85%).
3. **The Fatal Flaw of TF-IDF & Honest Disclosure on `theft_lost_legal`:** While TF-IDF won on headline accuracy, **it achieved 0/18 correct (0.00% recall, 0.00% F1) on safety-critical tickets (`theft_lost_legal`)**, classifying every police report, stolen iPhone, and consumer court threat as `other_unclear`. In an enterprise support environment, deploying TF-IDF would lead to catastrophic legal and brand liability. The LLM, by contrast, achieved **11/18 correct with 100.0% precision (11 TP, 0 FP, 7 FN; F1 = 0.76, Recall = 61.1%)**. Stating the raw count (11/18 correct) prevents overstating precision: the LLM intercepted 61.1% of high-risk cases on semantic understanding alone, and our hybrid escalation engine's deterministic keyword and confidence fallbacks successfully intercepted the remaining cases, achieving **100.00% safety recall (21/21)**.

---

### 3.2 Escalation Decision Engine Performance (N = 162)

The Hybrid Escalation Engine was benchmarked across the complete golden set ($N=162$, where 21 cases required mandatory escalation: stolen devices, legal disputes, harassment, and severe account compromises):

| Metric | Score | Operational Significance |
| :--- | :---: | :--- |
| **Escalation Recall** | **100.00%** | **Zero safety false negatives**: Successfully intercepted 21/21 high-risk queries (stolen hardware, police reports, consumer court lawsuits, abusive rants, account takeovers). |
| **Escalation Accuracy** | **51.85%** | 84 out of 162 routing decisions matched ground truth. Reflects deliberate bias towards defensive over-escalation. |
| **Escalation Precision** | **21.21%** | 21 true escalations out of 99 total escalations routed to humans (78 false alarms). |
| **Escalation F1-Score** | **35.00%** | Mathematical reflection of the asymmetric trade-off: safety recall is maximized at the expense of autonomous deflection. |

**Confusion Breakdown ($N=162$):** True Positives: 21 | False Negatives: 0 | True Negatives: 63 | False Positives: 78.
All 78 false alarms occurred on ambiguous, high-frustration queries (e.g. *"somebody hacked my Apple account"*, *"store never gave me back my power cord"*, or classifier confidence $<0.65$). In enterprise customer support operations, an unnecessary human escalation incurs modest staffing cost; a missed legal threat, police report, or uncontained account takeover is a catastrophic failure.

---

### 3.3 Reply Quality (LLM-as-Judge with Independent Rubric, N = 35)

Per PRD §4.7 and Deliverable 3, $N=35$ generated draft replies were evaluated across 4 standardized rubric dimensions (1–5 integer scale) using an independent judge model (`nvidia/nemotron-3-super-120b-a12b:free` hosted on OpenRouter with local LM Studio verification):

| Rubric Dimension | Score (1–5) | Operational Meaning & Rubric Standard |
| :--- | :---: | :--- |
| **Groundedness** | **4.77 / 5.0** | Strict fidelity to retrieved precedents; zero fabricated URLs (`support.apple.com` or `apple.co` only) and no unverified replacement promises. |
| **Correctness** | **4.66 / 5.0** | Accurately targets the core customer complaint with appropriate diagnostic triage questions. |
| **Tone Match** | **4.86 / 5.0** | Exemplary Apple Support persona: calm, empathetic, professional, within Twitter's 240-character limit. |
| **Resolution Likelihood** | **4.60 / 5.0** | Provides clear, actionable next steps (DM link, diagnostic settings path, or official KB guide). |
| **Overall Quality Mean** | **4.72 / 5.0** | Indicates high production-readiness for human-in-the-loop agent assist. |

#### Judge Model Identity & Scoring Inconsistency Analysis
1. **Judge Model Identity:** All reported judge ratings were produced by **NVIDIA Nemotron 3 Super 120B** (`nvidia/nemotron-3-super-120b-a12b:free`) accessed via OpenRouter, with offline/local fallback support via `google/gemma-4-e4b`. Per PRD §2.1 and §4.7, using an independent, high-capacity model family (120B Nemotron evaluating drafts generated by 4B Gemma) strictly eliminates self-preference bias.
2. **Deprecation of Earlier Candidate (`gemini-2.0-flash-lite`):** During initial prototyping, Google's `gemini-2.0-flash-lite:free` was evaluated as a judge. However, it was subsequently retired and made unavailable on OpenRouter's free tier. This volatility exemplifies the fragility of relying on external free-tier endpoints and reinforces why pinned model identifiers and SHA-256 response caching (`data/cache.sqlite`) are mandatory for reproducible evaluation.
3. **Risk of Scoring Inconsistency from Judge Rotation:** Different judge models possess fundamentally different calibration baselines and leniency thresholds (e.g., Nemotron 120B vs. Gemma 4B vs. Gemini Flash). If evaluation calls dynamically rotate across different judge models mid-benchmark due to rate limits or network fallbacks, inter-judge leniency variance injects severe confounding noise into the scores, corrupting both Cohen's $\kappa$ and Pearson $r$ correlation against human ground truth. To prevent this confounding effect, all headline judge evaluations and human-alignment pairs were evaluated against the single pinned Nemotron 120B model with deterministic SQLite caching.

---

### 3.4 Evidence of Judge-Human Agreement (N = 35 Paired Comparisons)

To satisfy assignment Deliverable 3 and the PRD's 30–50 example requirement, $N=35$ generated replies were independently hand-scored by a human reviewer on the identical 1–5 rubric. Statistical alignment between human and judge ratings:

- **Pearson Correlation ($r$):** **0.45** ($p = 0.007$, $N = 35$, statistically significant positive linear correlation).
- **Spearman Rank Correlation ($\rho$):** **0.26** ($N = 35$, monotonic rank-order alignment).
- **Quadratic Weighted Cohen's Kappa ($\kappa$):** **0.11** ($N = 35$, captures exact categorical bin agreement).
- **Mean Absolute Error (MAE):** **0.67 points** (Human mean: 4.12, Judge mean: 4.72 on a 5-point scale).

#### Statistical Sample Size & Defensible Precision Disclosure
In earlier rapid prototyping on a preliminary 14-pair slice, Pearson $r$ was calculated at 0.8219 ($p = 0.0003$). However, reporting Pearson correlation to four decimal places on 14 data points overstates the precision that the sample size can actually support. Expanding the evaluation to $N=35$ (matching the PRD target of 30–50 hand-scored examples) provides a substantially more defensible, realistic empirical assessment:
1. **Stabilized Correlation ($r = 0.45$, $p < 0.01$):** At $N=35$, the correlation remains statistically significant ($p = 0.007$), confirming positive alignment while revealing the true empirical variance between human evaluators and automated LLM judges. Reporting $r = 0.45$ (two decimal places) appropriately reflects the statistical confidence interval ($95\%\ \text{CI} \approx [0.14, 0.68]$).
2. **The "Kappa Paradox" ($\kappa = 0.11$ vs. $r = 0.45$):** A Cohen's $\kappa$ of 0.11 reflects the well-documented **Feinstein & Cicchetti (1990) Kappa Paradox**:
   - **Severe Marginal Ceiling Skew:** High-quality support replies are heavily concentrated in scores of 4 and 5 (Human mean: 4.12, Judge mean: 4.72). When marginal prevalence is heavily skewed, expected chance agreement $P_e$ becomes exceptionally high ($\approx 0.70$). Because $\kappa = \frac{P_o - P_e}{1 - P_e}$, a high $P_e$ severely penalizes $\kappa$ even when observed agreement is high.
   - **Systematic Judge Leniency Bias (+0.60 MAE):** The LLM judge exhibited a consistent positive calibration bias of +0.60 points, frequently awarding 5/5 to replies that a human evaluator rated 4/5. Because integer-binned $\kappa$ requires exact category matches (4 vs. 5), this consistent 1-point shift suppresses $\kappa$, whereas MAE confirms that judge scores deviate by less than 0.7 points on average.

---

## 4. Failure Analysis: Top 5 Failure Modes

By inspecting the misclassifications and routing discrepancies across all 162 evaluation cases in `agent_eval_outputs.json`, we identified the top 5 concrete failure modes:

### Failure Mode 1: Multi-Intent Composite Queries (Security vs. App Service)
- **Verifiable Example (Eval ID 5):** *"@AppleSupport I am trying to reset my security questions and it is not allowing me to. I am now locked out of iMessage"*
- **Ground Truth:** `apps_services_media` | **Model Prediction:** `account_icloud_security` | **Action:** Escalated (`llm_reasoner`).
- **Hypothesis:** Customer inquiries frequently mention an account trigger ("security questions") that causes an application failure ("locked out of iMessage"). Single-label classification forces an arbitrary choice. The attention mechanism keyed on the security phrasing rather than the service symptom.
- **Production Fix:** Implement multi-label classification or hierarchical intent decomposition (`primary_domain: account`, `impacted_service: imessage`).

### Failure Mode 2: Over-Escalation on Security Phishing / Scam Suspicion
- **Verifiable Example (Eval ID 28):** *"@AppleSupport Please, this alert doesn’t tell me what I’m giving my password to or what for and it bothers me constantly. How am I supposed to know some third party app isn’t trying to scam me? https://t.co/qm5kVavhqt"*
- **Ground Truth:** Auto-handle (Provide standard phishing advisory link) | **System Action:** Escalated (`deterministic_rule: fraud/scam keyword`).
- **Hypothesis:** Hardcoded regex rules for words like "scam" and "fraud" fired conservatively. While safe, informational inquiries about verifying alert legitimacy do not require immediate human specialist intervention.
- **Production Fix:** Add an informational regex whitelist for "is this a scam / legit" queries to auto-respond with Apple's standard security guidance (`apple.co/reportphishing`).

### Failure Mode 3: Emotional Rants Obscuring Core Technical Symptoms
- **Verifiable Example (Eval ID 38):** *"Disappointing service from @115858. Covent Garden “genius” - rude, patronising and unhelpful. My location and WiFi can still not be found!"*
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

While our system achieves **56.17% intent classification accuracy** (vs. 62.96% for TF-IDF), **100.00% safety escalation recall** (with 51.85% escalation accuracy), and an **average reply quality of 4.72 / 5.0**, reporting these figures without context would be intellectually dishonest:

1. **Resolution of Sample Size Limitations (Promoted to Full N=162 Golden Set):**
   Initial prototyping relied on an $N=36$ subsample to navigate cloud API rate limits. However, at $N=36$, a single flipped label changed accuracy by 2.78%, introducing unquantifiable sampling noise. By migrating inference to locally hosted LM Studio with Apple Silicon Metal acceleration, **all headline evaluation metrics were re-run on the complete 162-item golden evaluation set** (18 verified examples per class across 9 categories). Similarly, LLM-as-judge scoring was expanded from an initial 14-pair spot-check to $N=35$ ground-truth paired comparisons, meeting the PRD's 30–50 example target and reporting correlation at defensible precision ($r = 0.45$, $p < 0.01$).

2. **The Baseline Paradox — Accuracy Does Not Equal Safety:**
   As demonstrated in §3.1, a simple TF-IDF linear classifier beats our Few-Shot LLM by 6.79% on headline accuracy (62.96% vs. 56.17%). However, reporting TF-IDF as the "better" support model would be catastrophically misleading: **TF-IDF achieved 0/18 correct (0.00% recall) on legal threats and stolen hardware**. In customer support, an agent that correctly labels battery complaints but routes lawsuits to a chatbot is unusable. The LLM's value lies not in raw lexical memorization, but in the semantic safety backbone that enabled **100% safety recall (21/21)**.

3. **Per-Class Sample Sizes Must Accompany F1 Percentages:**
   Our Few-Shot LLM achieved a **0.76 F1** on `theft_lost_legal`. Reporting "76% F1" without context implies continuous precision that small samples cannot support. We explicitly state the raw numbers alongside percentages: the LLM got **11/18 correct with 100% precision (11 TP, 0 FP, 7 FN)**. Disclosing exact counts ensures transparent evaluation of rare, high-stakes intents.

4. **Local Gemma 4B Parametric Limits & Category Boundary Bleed:**
   The headline 56.17% classification accuracy was achieved with a 4-billion parameter open-weights model (`google/gemma-4-e4b`) hosted locally. A 4B parameter model has substantially lower parametric reasoning capacity than frontier models (e.g. GPT-4o, Claude 3.5 Sonnet). This constrained capacity is directly responsible for category boundary bleed (e.g., over-predicting `os_software_glitch` on generic app complaints, where recall reached 83.3% but precision fell to 36.6%).

5. **The Resolution Heuristic is an Imperfect Operational Proxy:**
   We labeled historical threads as "resolved" if the customer did not tweet back after Apple's reply. In reality, a customer often ceases replying because they gave up in frustration, switched to phone support, or visited an Apple Store in person. The Resolution KB inevitably includes instances where AppleSupport provided an unhelpful response, yet the absence of a customer follow-up caused our RAG system to treat it as a gold-standard precedent.

6. **Stratified Sampling Masks Production Volume Skew:**
   In real production support, $>50\%$ of incoming tweets are generic chatter (`other_unclear`) or recurring iOS update rants. Our evaluation set deliberately balanced all categories to exactly 18 examples per class. While this stratification is essential for measuring rare-class recall (such as theft and legal disputes), it means our 56.17% accuracy does **not** reflect real-world unstratified throughput, where performance on noisy general chatter dominates operational metrics.

7. **The Judge Agreement "Kappa Paradox" & Inter-Judge Leniency Inconsistency:**
   Reporting only Pearson correlation ($r = 0.45$) would overstate judge agreement by hiding the low Cohen's kappa ($\kappa = 0.11$). As analyzed in §3.4, the combination of score range restriction (most replies score 4 or 5) and a +0.60 MAE leniency bias compresses $\kappa$. Furthermore, different judge models have distinct leniency baselines (e.g., Nemotron 120B vs. Gemma 4B vs. Gemini Flash). Pinned caching on a single judge model (`nvidia/nemotron-3-super-120b-a12b:free`) was required to prevent cross-model distortion.

8. **Escalation Accuracy (51.85%) Conceals Extreme Safety-Bias:**
   An escalation accuracy of 51.85% sounds mediocre until decomposed into its constituent metrics: **100.00% Recall vs. 21.21% Precision**. The escalation engine produced 78 false alarms out of 162 queries, intentionally trading away human deflection efficiency to ensure zero safety leakage. Framing this as a "48.15% error rate" would misrepresent a deliberate, defensive architectural choice.

9. **Historical Dataset Drift:**
   The dataset captures Twitter support in late 2017 (iOS 11, iPhone X launch, macOS High Sierra). Many referenced support URLs (`apple.co/...`) and specific workarounds (e.g. the iOS 11 'i' keyboard autocorrect glitch) are obsolete today. Headline numbers demonstrate architectural soundness on 2017 data, but production deployment on modern iOS 18 requires re-indexing current support documentation.

10. **Free-Tier Infrastructure Volatility & Model Deprecation:**
   Free-tier API endpoints on aggregators like OpenRouter suffer from rapid model turnover. For instance, `gemini-2.0-flash-lite:free` was used during early prototyping but was subsequently retired from the free tier, requiring migration to `nvidia/nemotron-3-super-120b-a12b:free`. In production, relying on unpinned free-tier endpoints without fallback guarantees or response caching introduces operational fragility.

---

## 6. What We Would Do Next with One More Week

1. **Multi-Turn Context Tracking:** Extend the agent beyond single-turn replies to maintain conversational memory across multi-tweet customer threads using session state.
2. **Active Learning on Ambiguous Disagreements:** Automatically pipe examples where classifier confidence is borderline ($0.50 - 0.65$) into a human review queue to continually fine-tune the classification prompt.
3. **Dynamic Support KB Scraping:** Replace static tweet precedents with live ingestion of official Apple Support articles (`support.apple.com/HT...`) to ground replies in authoritative technical documentation rather than Twitter agent summaries.
4. **A/B Testing Against Paid Frontier Models:** Benchmark our free-tier rotation pipeline against paid frontier models (Claude 3.5 Sonnet, GPT-4o) to quantify the exact ROI and quality delta of paid APIs vs zero-cost local inference.
5. **Real-time Streaming Webhooks:** Package the pipeline into a FastAPI microservice with Twitter/Webhook event listeners for automated live shadow-testing.
