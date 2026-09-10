"""Curates the initial 160 stratified Golden Evaluation Set candidates strictly from
the held-out evaluation pool (eval_pool.jsonl).

Ensures zero data leakage between Resolution KB and Evaluation set.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import random
from collections import defaultdict, Counter
from src.taxonomy.taxonomy_def import TAXONOMY, INTENT_NAMES

random.seed(42)

EVAL_POOL_PATH = "data/processed/eval_pool.jsonl"
GOLDEN_OUTPUT_PATH = "golden_set/golden_eval_set.json"

# High-precision seed keyword matchers for stratified sampling from held-out pool
INTENT_FILTERS = {
    "battery_power_issue": [
        r'\bbattery\b', r'\bdrain\w*', r'\bcharging\b', r'\bcharge\b', r'\bdies\b', r'\bdying\b', r'\bpercentage\b'
    ],
    "os_software_glitch": [
        r'\bglitch\w*', r'\bupdate\b', r'\bfrozen\b', r'\bfreez\w*', r'\bkeyboard\b', r'\bautocorrect\b', r'\bboot\b', r'“I️”', r'\bletter i\b'
    ],
    "apps_services_media": [
        r'\bmusic\b', r'\bitunes\b', r'\bapp store\b', r'\bimessage\b', r'\bfacetime\b', r'\bsafari\b', r'\bplaylist\b', r'\bdownload app\b'
    ],
    "connectivity_network": [
        r'\bwi-?fi\b', r'\bbluetooth\b', r'\bcellular\b', r'\bairpods\b', r'\bconnect\w*', r'\bdisconnect\w*', r'\bno service\b'
    ],
    "account_icloud_security": [
        r'\bapple id\b', r'\bicloud\b', r'\bpassword\b', r'\block\w*', r'\b2fa\b', r'\bverification code\b', r'\bphishing\b', r'\bhacked\b'
    ],
    "hardware_screen_audio": [
        r'\bscreen\b', r'\bdisplay\b', r'\bspeaker\b', r'\bmic\b', r'\bmicrophone\b', r'\bcrack\w*', r'\bbutton\b', r'\bwater\b', r'\bshattered\b'
    ],
    "orders_repairs_store": [
        r'\border\b', r'\bship\w*', r'\btracking\b', r'\bstore\b', r'\bgenius bar\b', r'\brepair\b', r'\btrade-?in\b', r'\breplacement\b'
    ],
    "theft_lost_legal": [
        r'\bstolen\b', r'\btheft\b', r'\bpolice\b', r'\blawsuit\b', r'\bsue\b', r'\blegal\b', r'\blawyer\b', r'\bthief\b', r'\brobbed\b'
    ]
}

def clean_text(text: str) -> str:
    t = re.sub(r'@\w+', '', text)
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def curate_golden_candidates(target_per_intent: int = 18):
    os.makedirs("golden_set", exist_ok=True)
    print(f"Reading held-out evaluation pool from {EVAL_POOL_PATH}...")
    
    intent_buckets = defaultdict(list)
    other_bucket = []
    
    with open(EVAL_POOL_PATH) as f:
        for line in f:
            t = json.loads(line)
            raw = t["customer_text"]
            cleaned = clean_text(raw)
            if len(cleaned) < 25 or len(cleaned) > 280:
                continue
                
            matched_intent = None
            # Check intent matchers
            for intent, patterns in INTENT_FILTERS.items():
                for pat in patterns:
                    if re.search(pat, cleaned, re.IGNORECASE):
                        matched_intent = intent
                        break
                if matched_intent:
                    break
                    
            if matched_intent:
                intent_buckets[matched_intent].append(t)
            else:
                other_bucket.append(t)
                
    print("Found candidate counts per intent in held-out slice:")
    for intent in INTENT_NAMES:
        count = len(intent_buckets[intent]) if intent != "other_unclear" else len(other_bucket)
        print(f"  - {intent:<25}: {count:,} candidates")
        
    golden_set = []
    eval_id = 1
    
    for intent in INTENT_NAMES:
        candidates = intent_buckets[intent] if intent != "other_unclear" else other_bucket
        # Target 16-20 per intent
        k = min(target_per_intent, len(candidates))
        chosen = random.sample(candidates, k)
        
        for item in chosen:
            raw_text = item["customer_text"]
            clean_t = clean_text(raw_text)
            
            # Determine ground-truth escalation & reasoning
            if intent == "theft_lost_legal":
                escalate = True
                reason = "Legal threat, stolen device, or police involvement requires immediate human/specialist handling."
                ideal_reply = "Express empathy, direct to law enforcement / official Apple legal/police protocol, and escalate to safety team."
            elif intent in ["account_icloud_security", "hardware_screen_audio"]:
                # 50% escalate depending on severity
                if any(w in clean_t.lower() for w in ["stolen", "lawyer", "refund", "broken", "cracked", "locked"]):
                    escalate = True
                    reason = "Account lockout or physical hardware repair requires authenticated specialist or Genius Bar appointment."
                    ideal_reply = "Direct to Apple Support portal for identity verification or Genius Bar reservation."
                else:
                    escalate = False
                    reason = "Standard troubleshooting guide and diagnostic triage questions apply."
                    ideal_reply = "Provide diagnostic steps and official Apple Support KB link."
            else:
                escalate = False
                reason = "Technical support query eligible for auto-handling via historical resolution pattern."
                ideal_reply = f"Acknowledge {intent.replace('_', ' ')}, ask diagnostic clarifying questions, and share relevant Apple Support guide link."
                
            golden_set.append({
                "eval_id": eval_id,
                "tweet_id": item["customer_tweet_id"],
                "customer_text": raw_text,
                "clean_text": clean_t,
                "golden_intent": intent,
                "golden_escalate": escalate,
                "golden_escalation_reason": reason,
                "ideal_reply_criteria": ideal_reply,
                "actual_brand_reply": item.get("brand_reply_text", "")
            })
            eval_id += 1
            
    random.shuffle(golden_set)
    # Re-index
    for idx, item in enumerate(golden_set, 1):
        item["eval_id"] = idx
        
    with open(GOLDEN_OUTPUT_PATH, "w") as f:
        json.dump(golden_set, f, indent=2)
        
    print(f"\nSuccessfully generated Golden Evaluation Set: {GOLDEN_OUTPUT_PATH}")
    print(f"Total labeled examples: {len(golden_set)}")
    
    # Class breakdown
    class_dist = Counter(item["golden_intent"] for item in golden_set)
    print("\nGolden Set Class Distribution:")
    for intent, count in sorted(class_dist.items()):
        esc_count = sum(1 for x in golden_set if x["golden_intent"] == intent and x["golden_escalate"])
        print(f"  - {intent:<25}: {count:2d} examples ({esc_count:2d} escalate)")

if __name__ == "__main__":
    curate_golden_candidates()
