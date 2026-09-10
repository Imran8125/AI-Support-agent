"""Few-Shot LLM Intent Classifier for AppleSupport.

Uses LLMClient (LM Studio local server or OpenRouter free-tier rotation).
Features:
- Few-shot prompt with taxonomy definitions & canonical exemplars.
- Structured JSON output: {intent: str, confidence: float, reasoning: str}.
- Prompt hash caching in SQLite (re-runs take 0ms / 0 credits).
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
from src.taxonomy.taxonomy_def import TAXONOMY, INTENT_NAMES
from src.utils.llm_client import LLMClient

SYSTEM_PROMPT = """You are an expert customer intent classifier for Apple Support on Twitter.
Your job is to classify an incoming customer tweet into EXACTLY ONE of the following 9 predefined intents:

1. battery_power_issue: Fast battery drain, phone dying, charging problems, battery health drop.
2. os_software_glitch: iOS update errors, keyboard typing bugs (e.g. 'i' autocorrect to A?), app crashes, freezing, boot loops.
3. apps_services_media: Apple Music library erased/syncing, App Store downloads failing, iMessage/FaceTime activation errors, Safari issues.
4. connectivity_network: Wi-Fi dropping/refusing to connect, Bluetooth pairing errors (AirPods/car), cellular data signal loss.
5. account_icloud_security: Apple ID locked for security, 2FA code not received, iCloud backup errors, suspicious phishing emails.
6. hardware_screen_audio: Broken touchscreen, screen black/lines, speaker crackling, mic failure, water damage.
7. orders_repairs_store: Genius Bar reservations, repair status, order shipment delays (iPhone X shipping), trade-in values.
8. theft_lost_legal: Stolen Mac/iPhone, tracking stolen device with police, harassment, legal/lawsuit threats.
9. other_unclear: Vague rants without technical details, non-English tweets, emojis only, or non-actionable chatter.

You must respond ONLY with a valid JSON object in this exact format:
{
  "intent": "<exact_intent_name>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short 1-sentence explanation>"
}"""

FEW_SHOT_PROMPT_TEMPLATE = """Here are examples of how to classify tweets:

Customer: "My battery is draining so fast since iOS 11.1, barely lasts 3 hours."
Output: {{"intent": "battery_power_issue", "confidence": 0.98, "reasoning": "Complains of rapid battery discharge."}}

Customer: "your newest update glitch of replacing the 'I' with 'A?' is driving me insane"
Output: {{"intent": "os_software_glitch", "confidence": 0.99, "reasoning": "Refers to keyboard autocorrect software bug."}}

Customer: "Why did all my Apple Music songs disappear after I updated?"
Output: {{"intent": "apps_services_media", "confidence": 0.95, "reasoning": "Apple Music playlist/library synchronization issue."}}

Customer: "My phone keeps dropping wifi every 5 minutes while other devices stay connected."
Output: {{"intent": "connectivity_network", "confidence": 0.96, "reasoning": "Wi-Fi connection stability issue on the device."}}

Customer: "Somebody hacked my Apple account and Apple ID is locked, won't let me sign in."
Output: {{"intent": "account_icloud_security", "confidence": 0.98, "reasoning": "Compromised account, security lockout, or password reset."}}

Customer: "My iPhone speaker crackles during calls and screen has green vertical lines."
Output: {{"intent": "hardware_screen_audio", "confidence": 0.97, "reasoning": "Physical hardware damage affecting speaker and display."}}

Customer: "Ordered my iPhone X two weeks ago and tracking hasn't updated, when does it ship?"
Output: {{"intent": "orders_repairs_store", "confidence": 0.95, "reasoning": "Online store order delivery status inquiry."}}

Customer: "My MacBook was stolen at the airport, police report filed, can you track the serial?"
Output: {{"intent": "theft_lost_legal", "confidence": 0.99, "reasoning": "Stolen hardware involving police report."}}

Customer: "Apple is literally the worst company on earth smh 🙄"
Output: {{"intent": "other_unclear", "confidence": 0.90, "reasoning": "General complaint without specific technical detail."}}

Now classify the following incoming customer tweet:
"{customer_text}"
Output:"""

class LLMClassifier:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()

    def classify(self, text: str) -> dict:
        prompt = FEW_SHOT_PROMPT_TEMPLATE.format(customer_text=text)
        raw_resp, model_used = self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=150
        )
        
        # Parse JSON
        parsed = self._extract_json(raw_resp)
        if not parsed or parsed.get("intent") not in INTENT_NAMES:
            # Fallback to closest matching intent name if raw response had extra text
            matched_intent = "other_unclear"
            for name in INTENT_NAMES:
                if name in raw_resp.lower():
                    matched_intent = name
                    break
            parsed = {
                "intent": matched_intent,
                "confidence": 0.5,
                "reasoning": "Parsed via fallback regex extractor."
            }
            
        parsed["model_used"] = model_used
        return parsed

    def _extract_json(self, text: str) -> dict | None:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None


def run_llm_classification_eval(limit: int = 162):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    with open("golden_set/golden_eval_set.json") as f:
        golden_set = json.load(f)[:limit]
        
    print(f"Running Few-Shot LLM Intent Classifier on {len(golden_set)} golden examples...")
    classifier = LLMClassifier()
    
    eval_labels = []
    pred_labels = []
    predictions = []
    
    for idx, item in enumerate(golden_set, 1):
        txt = item["customer_text"]
        true_intent = item["golden_intent"]
        res = classifier.classify(txt)
        
        pred_intent = res["intent"]
        eval_labels.append(true_intent)
        pred_labels.append(pred_intent)
        
        predictions.append({
            "eval_id": item["eval_id"],
            "tweet_id": item["tweet_id"],
            "text": txt,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "confidence": res.get("confidence", 0.0),
            "reasoning": res.get("reasoning", ""),
            "model_used": res.get("model_used", ""),
            "correct": true_intent == pred_intent
        })
        
        status_char = "✓" if true_intent == pred_intent else "✗"
        if idx % 15 == 0 or idx == len(golden_set):
            curr_acc = accuracy_score(eval_labels, pred_labels)
            print(f"  [{idx}/{len(golden_set)}] Acc so far: {curr_acc*100:.1f}% | Last: {status_char} (True: {true_intent} -> Pred: {pred_intent})")
            
    acc = accuracy_score(eval_labels, pred_labels)
    f1 = f1_score(eval_labels, pred_labels, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("FEW-SHOT LLM CLASSIFIER RESULTS")
    print("="*50)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Macro-F1: {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(eval_labels, pred_labels, zero_division=0))
    
    # Save predictions
    out_path = "data/eval_results/llm_classifier_predictions.json"
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "predictions": predictions
        }, f, indent=2)
        
    print(f"Predictions saved to {out_path}")
    return acc, f1

if __name__ == "__main__":
    run_llm_classification_eval()
