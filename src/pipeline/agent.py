"""Unified AppleSupport AI Support Agent.

Integrates:
  1. Intent Classifier (Few-shot LLM with caching & fallbacks)
  2. RAG Reply Drafter (Cosine retrieval over Resolution KB)
  3. Hybrid Escalation Engine (Hard rules + soft signals + LLM reasoner)
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.classify.llm_classifier import LLMClassifier
from src.draft.rag_drafter import RAGReplyDrafter
from src.escalate.engine import EscalationEngine
from src.utils.llm_client import LLMClient

class AppleSupportAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()
        self.classifier = LLMClassifier(self.client)
        self.drafter = RAGReplyDrafter(self.client)
        self.escalator = EscalationEngine(self.client)

    def process_message(self, customer_text: str) -> dict:
        # 1. Classify intent
        cls_result = self.classifier.classify(customer_text)
        intent = cls_result["intent"]
        confidence = float(cls_result.get("confidence", 0.5))
        
        # 2. Draft grounded RAG reply
        draft_result = self.drafter.draft_reply(customer_text, intent=intent, k=3)
        top_sim = float(draft_result.get("top_similarity", 0.0))
        draft_reply = draft_result.get("draft_reply", "")
        
        # 3. Escalation Decision
        esc_result = self.escalator.decide(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence,
            retrieval_similarity=top_sim
        )
        
        return {
            "customer_text": customer_text,
            "intent": intent,
            "confidence": confidence,
            "intent_reasoning": cls_result.get("reasoning", ""),
            "draft_reply": draft_reply,
            "top_retrieval_similarity": top_sim,
            "precedents": draft_result.get("precedents", []),
            "escalate": esc_result["escalate"],
            "escalation_reason": esc_result["reason"],
            "escalation_method": esc_result["method"],
            "model_used": cls_result.get("model_used", "")
        }


if __name__ == "__main__":
    agent = AppleSupportAgent()
    query = "My battery is draining like crazy after the iOS 11.1 update, is there a fix?"
    print(f"\nProcessing query: \"{query}\"")
    output = agent.process_message(query)
    print("\n--- AGENT OUTPUT ---")
    print(f"Detected Intent:     {output['intent']} (confidence: {output['confidence']})")
    print(f"Drafted Reply:       {output['draft_reply']}")
    print(f"Escalate to Human:   {output['escalate']}")
    print(f"Escalation Reason:   {output['escalation_reason']}")
    print(f"Escalation Method:   {output['escalation_method']}")
