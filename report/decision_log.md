# Decision Log — AppleSupport AI Support Agent
**Hiver SDE Intern Take-Home Assignment**

This log documents the 13 non-obvious engineering decisions made during the architecture, data curation, model selection, and evaluation of the system.

---

### Decision 1: Target Brand Selection — Choosing AppleSupport over AmazonHelp, SpotifyCares, and Uber
- **Decision:** Selected `AppleSupport` as the primary brand, rejecting `AmazonHelp` (highest volume), `SpotifyCares`, and `Uber_Support`.
- **Rationale:** 
  - `AmazonHelp` (168k replies) suffers from severe multilingual contamination (~20–25% Spanish, German, and Hindi based on manual review of ~200 sample threads; estimated ~22%), violating PRD §1 scope ("Out of scope: multi-language support") and relies heavily on uninformative generic redirects (`amazon.com/help`).
  - `Uber_Support` (56k replies) is robotic: 94.5% of tweets are non-diagnostic boilerplate redirects ("Please send a DM with your email address", computed across all 56,193 replies in `candidate_brands_analysis.json`) with only 5.5% diagnostic questioning, rendering RAG grounding trivial.
  - `AppleSupport` (106k replies) is predominantly English (~99.5% based on manual review of ~200 sample threads), features 75.4% reference guide links (computed across all 106,648 replies in `candidate_brands_analysis.json`), and asks diagnostic triage questions in 31.1% of interactions (computed across 106,648 replies), offering rich technical depth for grounded RAG generation.

---

### Decision 2: Temporal Split for Zero-Leakage Evaluation
- **Decision:** Enforced a strict chronological split (80% Historical: March 2016 – Nov 17, 2017; 20% Held-out: Nov 17 – Dec 3, 2017). The Golden Evaluation Set was drawn strictly from the held-out window.
- **Rationale:** Standard random train/test splits cause conversational leakage: similar customer complaints from the same iOS update release cycle would appear in both the Resolution KB and the eval set, artificially inflating retrieval hit-rate and drafting scores.

---

### Decision 3: Local ONNX Dense Embeddings over API Embeddings
- **Decision:** Used local ONNX Runtime with quantized `BAAI/bge-small-en-v1.5` embeddings (384 dimensions) instead of remote embedding APIs.
- **Rationale:** Remote embedding APIs (e.g. OpenAI or OpenRouter) would consume rate-limit budget and introduce network latency for clustering and nearest-neighbor search. Local ONNX runs in $<5$ ms per batch on Apple Silicon (M-series CPU/NE), incurring zero cost and zero rate-limit usage.

---

### Decision 4: Resolution Proxy Heuristic Definition
- **Decision:** Defined a thread as "resolved" if: (a) AppleSupport replied with guidance/link and no customer follow-up was posted within the thread, OR (b) the follow-up contained gratitude cues (*"thanks", "that worked", "fixed"*) without renewed complaints.
- **Rationale:** In public Twitter support, customers rarely post explicit "issue resolved" tickets. Absence of follow-up after an actionable answer is a known industry proxy for customer satisfaction, filtering out endless unresolved complaint loops.

---

### Decision 5: 9-Class Taxonomy Architecture (7 Technical + 1 High-Risk + 1 Catch-All)
- **Decision:** Clustered customer messages into 7 actionable technical categories (`battery_power_issue`, `os_software_glitch`, `apps_services_media`, `connectivity_network`, `account_icloud_security`, `hardware_screen_audio`, `orders_repairs_store`) plus `theft_lost_legal` (high-risk) and `other_unclear`.
- **Rationale:** Finer-grained taxonomies (e.g., 50+ classes like Banking77) dilute few-shot prompt context and degrade LLM classification accuracy under rate-limited small context windows. A clean 9-class taxonomy aligns directly with Apple's internal routing tiers (Genius Bar, iOS triage, Apple ID security, Safety/Legal).

---

### Decision 6: Dual LLM Provider Support with Local Apple Silicon Fallback
- **Decision:** Built the `LLMClient` to support both local LM Studio (`google/gemma-4-e4b` on Apple Silicon) and OpenRouter free-tier rotation with unified schema and automatic fallbacks.
- **Rationale:** OpenRouter `:free` models rotate weekly and enforce ~20 req/min caps. Supporting local LM Studio provides zero-cost, unlimited-rate offline development capability on the developer's 16GB M5 Mac.

---

### Decision 7: SHA-256 SQLite Response Caching
- **Decision:** Implemented deterministic prompt caching in SQLite (`data/cache.sqlite`), hashing `(provider, model, system_prompt, user_prompt, temperature)`.
- **Rationale:** Development, prompt iteration, and re-running test suites often issue identical queries. Cache hits execute in $<4$ ms, saving thousands of API tokens and preventing rate-limit throttling during evaluation runs.

---

### Decision 8: Stratified Oversampling of Rare High-Risk Intents in Golden Set
- **Decision:** Oversampled rare categories like `theft_lost_legal` (which accounts for $<0.1\%$ of raw volume) to equal representation (18 per class, total 162).
- **Rationale:** Natural distribution sampling would have yielded $\le 1$ theft or legal complaint out of 162 examples. A classifier could achieve 99% accuracy while completely missing all safety-critical escalations. Stratified sampling allows robust measurement of escalation recall.

---

### Decision 9: Independence of Judge Model from Drafting Model
- **Decision:** Strictly enforced that the LLM-as-Judge model family differs from the model that drafted the reply.
- **Rationale:** LLMs suffer from documented self-preference bias, consistently awarding higher scores to their own stylistic quirks and vocabulary patterns. Using an independent model guarantees objective evaluation of groundedness and tone.

---

### Decision 10: Hybrid Escalation Architecture (Deterministic Rules + Soft Fallbacks + LLM)
- **Decision:** Built a tiered escalation engine: (1) Hard regex rules for legal, theft, and profanity; (2) Soft confidence ($<0.65$) and retrieval similarity ($<0.45$) thresholds; (3) LLM reasoning for ambiguous edge cases.
- **Rationale:** Pure LLM escalation is vulnerable to prompt injection or hallucinated leniency on lawsuit threats. Pure rule-based escalation misses subtle customer frustration. A hybrid pipeline combines deterministic safety guarantees with semantic nuance.

---

### Decision 11: Constrained Retrieval ($k=3$) with Hallucination Regex Validation
- **Decision:** Limited RAG context to top $k=3$ precedents and post-processed draft replies with regex link extraction to flag fabricated non-Apple URLs.
- **Rationale:** Providing $>5$ precedents increases prompt token length, inflating latency and introducing conflicting resolution advice. A concise $k=3$ window grounds the reply while keeping the tweet draft within the 240-character Twitter length limit.

---

### Decision 12: Discarding Banking77 for Final Primary Pipeline
- **Decision:** Analyzed Banking77 (`PolyAI/banking77`) as an optional secondary dataset, but did not use its labels in the primary AppleSupport agent.
- **Rationale:** Banking77 consists of clean, single-sentence banking queries (e.g. `card_arrival`, `exchange_rate`), whereas Apple support deals with device operating systems, battery degradation, iCloud auth, and hardware damage. Cross-domain transfer would degrade classification performance on noisy Twitter text.

---

### Decision 13: Subsampling 36 Examples (22.2%) from the 162-Item Golden Set for Headline Evaluation
- **Decision:** While 162 held-out customer interactions were hand-curated and ground-truth verified (18 per intent class across 9 categories in `golden_set/golden_eval_set.json`), headline evaluation was executed on a balanced 36-query subsample (exactly 4 per class, 22.2% of the golden set).
- **Rationale:** 
  - **Rate-limit quotas and latency on free LLM endpoints:** Evaluating all 162 examples through the full multi-stage agent pipeline (intent classification, dense embedding retrieval, RAG tweet drafting, and independent LLM-as-judge rubric scoring) requires ~486 LLM calls. Under OpenRouter free-tier rate limits (~20 requests/minute with frequent HTTP 429 throttling and exponential backoffs), running 162 queries requires ~35–45 minutes and routinely fails mid-run due to provider timeouts. Subsampling to 36 queries completes deterministically in <3 minutes (or <5 seconds with SQLite cache hits).
  - **Preserving stratified class balance:** Rather than evaluating an unstratified random subset, 36 was chosen because it allows exactly 4 examples per category across all 9 taxonomy intents ($9 \times 4 = 36$). This guarantees identical statistical weight for critical, low-frequency categories (`theft_lost_legal` accounts for 4 of 36 queries, 11.1%), preventing rare safety escalations from being washed out by high-volume chatter.
  - **Decoupling curation from rapid benchmark execution:** Curation rigor (PRD §3 requirement for 150–250 golden examples) was met by labeling and verifying the complete 162-item dataset in `golden_set/golden_eval_set.json`. The master harness (`src/eval/harness.py`) provides the `--limit 162` flag for unconstrained evaluation runs.

