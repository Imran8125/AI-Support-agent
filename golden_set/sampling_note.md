# Golden Evaluation Set — Sampling & Labeling Note
**Brand: AppleSupport | Total Size: 162 examples**

---

### 1. Sampling Methodology & Temporal Leakage Prevention

A critical failure mode in evaluation harnesses is **data leakage**, where examples from the evaluation set appear in the retrieval corpus or training data.

To strictly guarantee zero leakage:
1. All 106,625 reconstructed `AppleSupport` conversation threads were sorted chronologically by the customer's initial tweet timestamp.
2. The dataset was partitioned into two disjoint temporal slices:
   - **Historical KB / Training Pool (80%):** 85,300 threads spanning *March 4, 2016 to November 17, 2017 (12:21 PM)*. The Resolution Knowledge Base (RAG precedents) and training data for Baseline B are built exclusively from this pool.
   - **Held-out Evaluation Pool (20%):** 21,325 threads spanning *November 17, 2017 (12:22 PM) to December 3, 2017 (11:03 PM)*.
3. The Golden Evaluation Set of **162 examples** was sampled **exclusively from the held-out evaluation pool**. The system had never encountered these queries during KB construction.

---

### 2. Stratification Strategy

Real Twitter support volume is heavily skewed (e.g. general chatter and iOS update bugs dominate >50% of raw tweets). If sampled purely at random, rare but critical high-risk intents (such as legal threats, stolen hardware, or account lockouts) would constitute $<1\%$ of the eval set, rendering escalation recall statistically unmeasurable.

We employed **stratified sampling** with mild oversampling of rare classes:
- Target: Exactly **18 examples per class** across all 9 taxonomy categories.
- Total count: **162 ground-truth verified examples**.

| Intent Category | Count | Primary Problem Type | Ground Truth Escalation Policy |
| :--- | :---: | :--- | :--- |
| `battery_power_issue` | 18 | Rapid drain, dying at 20%, charging failure | Auto-handle (diagnostic triage + KB guide) |
| `os_software_glitch` | 18 | iOS 11 'i' autocorrect bug, freezing, boot loop | Auto-handle (reboot / update KB steps) |
| `apps_services_media` | 18 | Apple Music wipe, iMessage/FaceTime error | Auto-handle (re-sync / App Store troubleshooting) |
| `connectivity_network`| 18 | Wi-Fi dropped, Bluetooth AirPods disconnect | Auto-handle (Network settings reset guide) |
| `account_icloud_security`| 18 | Apple ID locked, 2FA code issue, phishing | Hybrid (Escalate if locked/phishing, else guide) |
| `hardware_screen_audio`| 18 | Screen black, crackling speaker, water damage | Hybrid (Escalate if hardware damaged $\rightarrow$ Genius Bar) |
| `orders_repairs_store` | 18 | iPhone X shipping delay, trade-in, repairs | Auto-handle (Store reservation / tracking policy) |
| `theft_lost_legal` | 18 | Stolen Mac, police complaint, lawsuit threats | **Mandatory Escalate (100% human agent)** |
| `other_unclear` | 18 | Non-English, vague rants, emojis | Auto-handle (Polite clarifying question) |

---

### 3. Labeling Rubric & Ground Truth Attributes

Each record in `golden_set/golden_eval_set.json` contains:
1. `tweet_id`: Original tweet identifier from Kaggle TWCS.
2. `customer_text`: The raw customer inquiry.
3. `golden_intent`: The single best-fitting intent category from our taxonomy.
4. `golden_escalate`: Boolean flag (`true` = escalate to human specialist; `false` = auto-handle with drafted reply).
5. `golden_escalation_reason`: Stated rationale justifying the decision (satisfies PRD §1.3 and assignment requirement).
6. `ideal_reply_criteria`: What a high-quality, grounded reply must contain (e.g., specific diagnostic question, acknowledgment, avoidance of unverified promises).
7. `actual_brand_reply`: The historical tweet drafted by AppleSupport agents for reference.

---

### 4. Golden Set Size & Comprehensive Evaluation Coverage (N = 162)

- **Complete Golden Evaluation Set ($N=162$):** 162 hand-curated and ground-truth verified records (exactly 18 per class across all 9 taxonomy categories) stored in `golden_set/golden_eval_set.json`.
- **Definitive Headline Benchmark Execution:** All reported headline evaluation metrics in `report/report.md` are executed across the complete 162-item set (100% of the golden set), eliminating small-sample variance. Execution is accelerated locally via Apple Silicon Metal acceleration on `google/gemma-4-e4b` in LM Studio, bypassing cloud rate limits.
- **Execution via CLI:** The master evaluation harness (`src/eval/harness.py`) defaults to evaluating all 162 items via `python -m src.eval.harness --limit 162 --judge-sample 35`.


