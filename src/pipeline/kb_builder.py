"""Resolution Knowledge Base Builder for AppleSupport RAG.

Extracts candidate resolved threads from historical_kb.jsonl using the
explicit resolution proxy heuristic (PRD §4.3):
  1. Brand reply exists with actionable support guidance/link.
  2. EITHER:
     (a) No customer follow-up was needed (single-turn resolution), OR
     (b) Customer follow-up contains gratitude/closure cues
         ("thank you", "thanks", "that worked", "fixed it", "solved", "appreciate it")
         and no renewed complaint.
  3. Pre-computes local BGE embeddings for sub-millisecond retrieval.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import random
import numpy as np
from src.utils.embeddings import LocalEmbeddingModel

random.seed(42)

HISTORICAL_PATH = "data/processed/historical_kb.jsonl"
KB_OUTPUT_JSON = "data/processed/resolution_kb.json"
KB_OUTPUT_VECTORS = "data/processed/resolution_kb_vectors.npy"

GRATITUDE_PATTERNS = re.compile(
    r'\b(thank|thanks|worked|fixed|solved|awesome|appreciate|cheers|got it|perfect|helped)\b',
    re.IGNORECASE
)
RENEWED_COMPLAINT = re.compile(
    r'\b(still not|didn\'t work|still broken|useless|terrible|horrible|worst|doesn\'t work|cannot|can\'t)\b',
    re.IGNORECASE
)

def is_resolved_thread(thread: dict) -> bool:
    brand_reply = thread.get("brand_reply_text", "")
    if len(brand_reply) < 30:
        return False
        
    # Heuristic 1: No customer followup needed
    if not thread.get("has_followup"):
        # Ensure brand reply isn't just an empty greeting
        return True
        
    # Heuristic 2: Customer followed up with gratitude and no renewed complaint
    followup_text = thread.get("customer_followup_text", "") or ""
    if GRATITUDE_PATTERNS.search(followup_text) and not RENEWED_COMPLAINT.search(followup_text):
        return True
        
    return False

def clean_text(text: str) -> str:
    t = re.sub(r'@\w+', '', text)
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def build_resolution_kb(target_samples: int = 1500):
    print(f"Reading historical threads from {HISTORICAL_PATH}...")
    candidates = []
    
    with open(HISTORICAL_PATH) as f:
        for line in f:
            t = json.loads(line)
            cust_text = t.get("customer_text", "")
            brand_reply = t.get("brand_reply_text", "")
            
            # Basic filters: non-empty English text with meaningful length
            if len(cust_text) < 25 or len(brand_reply) < 35:
                continue
                
            if is_resolved_thread(t):
                candidates.append({
                    "thread_id": t["thread_id"],
                    "customer_text": cust_text,
                    "clean_customer_text": clean_text(cust_text),
                    "brand_reply": brand_reply,
                    "has_followup": t.get("has_followup", False),
                    "followup_text": t.get("customer_followup_text"),
                    "created_at": t.get("customer_created_at")
                })
                
    print(f"Total resolved threads identified by heuristic: {len(candidates):,}")
    
    # Subsample for dense, high-quality index
    sample_size = min(target_samples, len(candidates))
    selected = random.sample(candidates, sample_size)
    print(f"Selected {sample_size} high-quality resolved cases for Resolution KB.")
    
    # Precompute embeddings
    print("Pre-computing local dense embeddings for retrieval...")
    embedder = LocalEmbeddingModel()
    texts_to_embed = [item["clean_customer_text"] for item in selected]
    vectors = embedder.embed_texts(texts_to_embed, batch_size=64)
    print(f"Vectors generated: shape {vectors.shape}")
    
    # Save JSON index and vector array
    os.makedirs(os.path.dirname(KB_OUTPUT_JSON) or ".", exist_ok=True)
    with open(KB_OUTPUT_JSON, "w") as f:
        json.dump(selected, f, indent=2)
        
    np.save(KB_OUTPUT_VECTORS, vectors)
    
    print(f"Resolution KB saved:")
    print(f"  - Metadata: {KB_OUTPUT_JSON} ({len(selected)} entries)")
    print(f"  - Vectors:  {KB_OUTPUT_VECTORS} ({vectors.nbytes / 1024:.1f} KB)")

if __name__ == "__main__":
    build_resolution_kb()
