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

### Decision 6: Primary Local LM Studio (Gemma 4B) Inference with Cloud OpenRouter Fallback & Judge Routing
- **Decision:** Built `LLMClient` with a primary local inference runtime (`google/gemma-4-e4b` via LM Studio on Apple Silicon) and OpenRouter free-tier API as an automated cloud fallback and dedicated host for the independent LLM-as-Judge (`nvidia/nemotron-3-super-120b-a12b:free`), coordinated via `LLM_PROVIDER=auto` (with explicit manual `lmstudio` / `openrouter` overrides) in `.env`.
- **Rationale:** 
  - **The Dev-Iteration Rate Limit Bottleneck:** The initial architecture was conceived entirely on OpenRouter free models, but strict rate limits (~20 req/min with frequent HTTP 429 throttling and exponential backoffs) made running a multi-stage agent pipeline (intent classification + RAG drafting + escalation reasoning) across hundreds of queries impractical at development iteration speed.
  - **Decoupled Workloads:** Pivoting to local LM Studio inference on Apple Silicon provided zero-cost, zero-latency, rate-limit-free execution for high-volume agent calls. Meanwhile, OpenRouter was strategically reserved for the independent LLM-as-Judge (`nvidia/nemotron-3-super-120b-a12b:free`), where independence from the drafting model (to strictly eliminate self-preference bias, PRD §2.1 & §4.7) matters far more than call volume.

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

### Decision 13: Transition from Rapid N=36 Prototyping Subsample to Full N=162 Golden Evaluation via Local GPU Acceleration
- **Decision:** While initial development utilized a balanced 36-query subsample (4/class, 22.2%) to navigate free-tier cloud rate limits during rapid iteration, all definitive headline benchmark metrics were promoted to the complete 162-item golden evaluation set (18 per class across all 9 taxonomy intents) executed locally via Apple Silicon Metal acceleration (`google/gemma-4-e4b` in LM Studio).
- **Rationale:** 
  - **Eliminating statistical sampling variance:** At $N=36$ across 9 classes (4 per class), a single misclassified query swayed accuracy by 2.78 percentage points, creating metric variance that obscured subtle model differences. Evaluating across the complete 162-item golden set provides a statistically robust, defensible foundation where each category has 18 verified ground-truth data points.
  - **Overcoming cloud rate limits with local hardware:** Executing all 162 queries through classification, retrieval, drafting, and escalation would have taken ~35–45 minutes on rate-limited free cloud endpoints due to rolling 20 req/min caps and exponential retry backoffs. Bringing up local LM Studio on Apple Silicon unified memory with Metal acceleration and multi-worker parallelism allowed the full 162-item evaluation (alongside $N=35$ judge evaluations) to execute deterministically without rate-limiting, network timeouts, or API costs.
  - **Transparent per-class reporting:** On the full 162-item set, we report raw counts (e.g., `11/18 correct` on `theft_lost_legal`) alongside percentages, preventing overstated precision on rare categories and proving that the LLM achieved 100% precision (11 TP, 0 FP, 7 FN) on safety-critical threats where TF-IDF scored 0/18 (0.0% recall).


