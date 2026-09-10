"""Hybrid Escalation Decision Engine for AppleSupport.

Combines:
  1. Deterministic hard rules (theft, legal, supervisor demand, harassment)
  2. Quantitative confidence & retrieval quality thresholds
  3. LLM-based reasoning for nuanced edge cases
Outputs structured decision: {escalate: bool, reason: str}
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
from src.escalate.rules import check_hard_escalation_rules
from src.utils.llm_client import LLMClient

CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_SIMILARITY_THRESHOLD = 0.45

LLM_ESCALATION_PROMPT = """You are an escalation supervisor for Apple Support.
Decide whether the following customer inquiry can be auto-handled by an AI assistant or must be escalated to a human support specialist.

Customer tweet: "{customer_text}"
Detected intent: {intent} (confidence: {confidence:.2f})
Best retrieval similarity from knowledge base: {similarity:.2f}

Guidelines:
- Auto-handle if it is a common technical issue (battery, software update bug, Wi-Fi, audio, app store) with clear known diagnostic steps.
- Escalate if it requires account authentication, private billing dispute, hardware repair inspection, lost/stolen device, legal threats, or expressed deep frustration.

Respond in strict JSON format:
{{
  "escalate": <true or false>,
  "reason": "<clear 1-sentence explanation of why it was auto-handled or escalated>"
}}"""

class EscalationEngine:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()

    def decide(
        self,
        customer_text: str,
        intent: str,
        confidence: float = 1.0,
        retrieval_similarity: float = 0.8
    ) -> dict:
        # Step 1: Check deterministic hard rules (zero API cost)
        hard_trigger, hard_reason = check_hard_escalation_rules(customer_text)
        if hard_trigger:
            return {
                "escalate": True,
                "reason": hard_reason,
                "method": "deterministic_rule"
            }
            
        # Step 2: High-risk intent policy
        if intent == "theft_lost_legal":
            return {
                "escalate": True,
                "reason": "Intent policy: Stolen device or legal issue requires specialist handling.",
                "method": "intent_policy"
            }
            
        # Step 3: Low confidence or low grounding soft fallback
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "escalate": True,
                "reason": f"Soft threshold: Classifier confidence ({confidence:.2f}) below safe threshold ({CONFIDENCE_THRESHOLD}).",
                "method": "low_confidence"
            }
            
        if retrieval_similarity < RETRIEVAL_SIMILARITY_THRESHOLD:
            return {
                "escalate": True,
                "reason": f"Soft threshold: Best knowledge base match similarity ({retrieval_similarity:.2f}) below grounding threshold ({RETRIEVAL_SIMILARITY_THRESHOLD}).",
                "method": "low_similarity"
            }

        # Step 4: LLM decision for remaining ambiguous cases
        prompt = LLM_ESCALATION_PROMPT.format(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence,
            similarity=retrieval_similarity
        )
        
        raw_resp, model_used = self.client.generate(
            prompt=prompt,
            system_prompt="You are an expert customer escalation decision engine for Apple.",
            temperature=0.1,
            max_tokens=100
        )
        
        # Parse JSON
        parsed = self._extract_json(raw_resp)
        if parsed and "escalate" in parsed:
            return {
                "escalate": bool(parsed["escalate"]),
                "reason": str(parsed.get("reason", "Decided by LLM reasoning.")),
                "method": f"llm_reasoner ({model_used})"
            }
            
        # Default fallback
        return {
            "escalate": False,
            "reason": "Standard inquiry eligible for auto-reply.",
            "method": "default_fallback"
        }

    def _extract_json(self, text: str) -> dict | None:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None


if __name__ == "__main__":
    engine = EscalationEngine()
    test_cases = [
        ("My MacBook was stolen yesterday, please help", "theft_lost_legal", 0.95, 0.75),
        ("I will sue Apple in court if this isn't fixed today", "other_unclear", 0.85, 0.60),
        ("How can I fix the letter i autocorrect bug in iOS 11?", "os_software_glitch", 0.98, 0.88),
        ("I want to speak to a human agent right now", "other_unclear", 0.90, 0.70)
    ]
    print("--- TESTING HYBRID ESCALATION ENGINE ---")
    for txt, intent, conf, sim in test_cases:
        res = engine.decide(txt, intent, conf, sim)
        print(f"\nQuery:    \"{txt}\"")
        print(f"Escalate: {res['escalate']} | Method: {res['method']}")
        print(f"Reason:   {res['reason']}")
