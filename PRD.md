# PRD — AI Support Agent for [Brand Name]
**Hiver SDE Intern Take-Home Assignment**

---

## 1. Objective

Build and prove out an AI system that, given an incoming customer tweet to one support brand, can:
1. Classify intent into a small, data-derived taxonomy
2. Draft a reply grounded in how the brand has historically resolved similar issues
3. Decide auto-handle vs. escalate-to-human, with a stated reason

The deliverable is not "a working demo" — it's **evidence that the system is good enough to trust**, via a golden eval set, a judged evaluation harness, and an honest account of the system's limits.

**Out of scope:** multi-language support, real account/order lookups, multi-turn dialogue management beyond the first reply, non-Twitter channels, real-time deployment.

---

## 2. Constraints (drives most of the design below)

- **Models: free-tier only, via OpenRouter.** `:free` model IDs on OpenRouter carry hard rate limits — commonly ~20 requests/minute and ~200 requests/day *per model*. This is the single biggest constraint on the whole pipeline and must be designed around, not discovered late.
- Free model availability/identity rotates (providers change weekly). **Do not hardcode a single model ID as a dependency.** Query `GET https://openrouter.ai/api/v1/models`, filter `pricing.prompt == "0"`, and pick at build time. As of this writing, credible free candidates include `openrouter/free` (auto-router across available free models), and individual `:free`-suffixed models from providers like Nex AGI, InclusionAI, Meta, Google, and Qwen — confirm current availability before finalizing.
- Time budget: ASAP, target 3–4 focused days.

### 2.1 Rate-limit budget math (do this before writing code)

Rough call inventory for one full pipeline run:
| Stage | Calls needed |
|---|---|
| Cluster labeling (taxonomy) | ~20–40 (one per cluster, iterate) |
| Golden set labeling assist (optional, human does final call) | 0 (do this yourself, not via API) |
| Classifier inference (golden set eval) | ~150–250 |
| Classifier inference (dev/debug iterations) | 200–500 |
| Reply drafting (golden set) | ~150–250 |
| Escalation reasoning (golden set) | ~150–250 |
| LLM-as-judge scoring (golden set × rubric dims) | ~450–1250 (3–5 dims × 150–250 examples) |
| **Total** | **~1,100–2,500+ calls** |

Against a ~200/day/model cap, this does **not** fit in one model in one day. Mitigations, in order of preference:
1. **Rotate across 3–4 different free models** for the high-volume stages (classification, drafting) — this also strengthens your report (you can compare free-model quality as a side finding).
2. **Cache every response** (prompt hash → response) so re-runs during debugging cost zero extra calls.
3. **Spread golden-set evaluation runs across 2 days** if needed — flag this plainly in the decision log rather than hiding it.
4. Keep the **judge model different from the generation model** (needed anyway to avoid self-preference bias) — this naturally spreads load across models too.
5. If still tight, reduce golden set to 150 (the deliverable's floor) rather than 250.

---

## 3. System Architecture

```
Raw dataset (Kaggle Twitter CSVs)
        │
        ▼
[1] Thread Reconstruction  ──► brand-filtered, threaded conversations
        │
        ▼
[2] Intent Taxonomy Builder ──► 6–10 labeled intents (offline, one-time)
        │
        ▼
[3] Resolution KB Builder ──► per-intent bank of heuristically "resolved" past replies
        │
        ├──────────────┬──────────────────┐
        ▼              ▼                  ▼
 [4] Intent        [5] Reply          [6] Escalation
   Classifier        Drafter (RAG)      Decision
        │              │                  │
        └──────┬───────┴──────────────────┘
               ▼
        Agent output: {intent, draft_reply, escalate: bool, reason}
               │
               ▼
        [7] Evaluation Harness (golden set, baselines, LLM-judge, agreement check)
```

---

## 4. Component Specs

### 4.1 Thread Reconstruction
- Input: Kaggle Twitter Customer Support CSV.
- Join on `in_response_to_tweet_id` to build `customer_msg → brand_reply → customer_followup` chains.
- Filter to one brand as responder (`author_id` == brand handle).
- Output schema per thread: `{thread_id, customer_msgs: [...], brand_reply: str, followup: str|None, timestamps}`.

### 4.2 Intent Taxonomy Builder (one-time, offline)
- Sample ~1,000–2,000 initial customer messages.
- Embed with a local/free embedding model (sentence-transformers `all-MiniLM-L6-v2` — runs locally, zero API calls, saves your rate-limit budget for the parts that need an LLM).
- Cluster (HDBSCAN, or k-means with a silhouette sweep over k=6..15).
- For each cluster, sample 5–8 messages and use **one** free-model call to propose a name + description.
- Manually merge/prune to **6–10 final intents** + an `other/unclear` bucket. Document merge decisions in the decision log.

### 4.3 Resolution KB Builder
- Define the resolution heuristic explicitly, e.g.: *"thread is resolved if there is a brand reply and either (a) no customer follow-up within the thread, or (b) the follow-up contains closure/gratitude language and no new complaint."*
- This is a proxy, not ground truth — flag it in the report's "misleading" section.
- Build a small per-intent index: `{intent → [{customer_msg, brand_reply, embedding}, ...]}` for retrieval.

### 4.4 Intent Classifier
- **Baseline A (trivial):** majority-class predictor.
- **Baseline B (simple):** TF-IDF + logistic regression, trained on a labeled subsample (you'll need to weak-label a training set using cluster assignments from 4.2, separate from the golden set).
- **System:** few-shot prompt to a free LLM — taxonomy + 2–3 examples per intent, ask for intent + confidence.
- Log every prediction with the model ID used (for the free-model rotation comparison).

### 4.5 Reply Drafter (RAG)
- Retrieve top-k (k=3–5) similar resolved cases from the Resolution KB (cosine similarity on the local embeddings — no API call needed for retrieval itself).
- Prompt template (concise, edit to taste):
  ```
  You are drafting a support reply for [Brand]. Customer message: "{msg}"
  Detected intent: {intent}
  Here is how this brand has resolved similar issues before:
  {retrieved_examples}
  Draft a reply consistent with this brand's tone and resolution pattern.
  Do NOT invent order numbers, refund amounts, or account-specific facts not given above.
  ```
- Track whether the draft cites something not present in retrieved context (a cheap regex/keyword check for numbers/amounts not in the prompt catches a lot of hallucination cases).

### 4.6 Escalation Decision (hybrid)
- **Hard rules** (no LLM needed, zero API cost): safety/legal/harassment keyword match, explicit "let me speak to a human/manager", repeated contact from same customer without resolution → force escalate.
- **Soft signal:** classifier confidence below threshold, retrieval similarity below threshold (nothing grounds the reply well), negative-sentiment score.
- **LLM call:** given intent + confidence + retrieval score + sentiment, output `{escalate: bool, reason: str}` in one short call — this satisfies "decide... with a stated reason."

### 4.7 Evaluation Harness
- **Classification:** accuracy, macro-F1 vs. golden labels, for trivial baseline, simple baseline, and system.
- **Escalation:** precision/recall vs. your golden escalate/auto-handle labels.
- **Reply quality (LLM-as-judge):** score groundedness, correctness, tone-match, resolution-likelihood (1–5 each) using a model **different from the drafting model**.
- **Judge agreement:** hand-score 30–50 replies yourself on the same rubric; report correlation or Cohen's kappa between your scores and the judge's. This is a required deliverable, not optional polish.
- **Retrieval quality (supporting metric):** hit-rate — of the top-k retrieved precedents, how many did you judge actually relevant, on a 30-example spot check.

---

## 5. Golden Evaluation Set (150–250 examples)

- Sample from a **held-out time slice** of threads not used to build the Resolution KB (prevents leakage).
- Stratify by the 6–10 intents from §4.2, oversampling rare intents slightly so they're not invisible in metrics.
- Label per example: correct intent, escalate/auto-handle judgment, and (subset of ~50) a note on what a good reply should contain.
- Keep a running sampling/labeling log as you go — this becomes the "short note on how you sampled and labeled" deliverable directly.

---

## 6. Report Mapping (max 6 pages)

| Report section | Where it comes from in this PRD |
|---|---|
| Problem framing / what you didn't build | §1 scope, plus your brand-choice reasoning |
| Results vs. 2 baselines | §4.4 (classifier), extend same pattern to escalation if time allows |
| Failure analysis (top 5) | Pull lowest-scoring golden examples from §4.7, write real hypotheses |
| "What's misleading about my headline number" | Resolution heuristic is a proxy (§4.3); golden set is small and self-labeled (§5); judge may favor fluency over correctness; free-model outputs may vary run-to-run due to provider load; dataset is dated, brand behavior may have shifted |
| What you'd do next | e.g., multi-turn dialogue handling, paid-model comparison, active learning on hard cases |

---

## 7. Decision Log — seed list (expand as you build)

1. Brand chosen and why (data volume + resolution clarity)
2. Resolution heuristic definition
3. Intent taxonomy size and merge decisions
4. Free-model rotation strategy and which model did which job
5. Judge model chosen ≠ drafting model, and why
6. Escalation rule thresholds (confidence, similarity, sentiment cutoffs)
7. Golden set sampling strategy (stratification, time-slice held out)
8. Retrieval k value
9. Caching strategy for rate-limit management
10. What was cut due to time/rate-limit constraints, and why

---

## 8. Suggested Repo Structure

```
/data/           raw + processed thread data (not committed, .gitignore)
/notebooks/      exploratory clustering, brand selection eyeballing
/src/
  pipeline/      thread reconstruction, KB builder
  taxonomy/      clustering + labeling scripts
  classify/      baselines + LLM classifier
  draft/         RAG reply drafter
  escalate/      hybrid escalation logic
  eval/          harness, judge, agreement calc
/golden_set/     your 150–250 labeled examples (this IS committed)
/report/         report.md, decision_log.md
README.md        setup + reproduce headline results in <15 min
```

---

## 9. Milestones (compressed timeline)

- **Day 1:** Thread reconstruction, brand selection (eyeball 200 threads first), taxonomy built.
- **Day 2:** Both baselines + LLM classifier; start golden set labeling in parallel.
- **Day 3:** Resolution KB, RAG drafter, escalation logic.
- **Day 4:** Eval harness, judge agreement check, failure analysis, report, decision log, README reproducibility pass.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Free-model rate limits stall the pipeline | Rotate models, cache aggressively, budget calls per §2.1 before coding |
| Free model quality too low/inconsistent for judge role | Spot-check judge outputs against your own 30–50 hand scores early, not at the end |
| Resolution heuristic mislabels many "resolved" threads | State the heuristic explicitly, discuss its failure modes in the report |
| Golden set skewed toward easy/common cases | Stratify by intent, deliberately include ambiguous/edge examples |
| README doesn't actually reproduce in 15 min | Test on a clean checkout/environment before submitting, not just your dev machine |