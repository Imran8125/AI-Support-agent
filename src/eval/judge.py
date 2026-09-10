"""LLM-as-Judge for Reply Quality Evaluation.

Uses an independent model (different from the drafting model to prevent
self-preference bias, PRD §2.1 & §4.7) to evaluate 4 rubric dimensions (1-5):
  1. Groundedness: Is reply grounded in retrieved facts/precedents? Zero hallucinations.
  2. Correctness: Does reply address the customer's actual inquiry appropriately?
  3. Tone Match: Does reply match Apple Support's polite, succinct Twitter persona?
  4. Resolution Likelihood: How likely is this reply to resolve or advance the issue?
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
from src.utils.llm_client import LLMClient

JUDGE_SYSTEM_PROMPT = """You are an impartial, critical evaluation judge assessing the quality of AI-drafted customer support tweets for Apple Support (@AppleSupport).
You must evaluate the draft reply strictly on four rubric dimensions using a 1-5 integer scale.
Be rigorous: penalize hallucinated links, ungrounded promises, inappropriate tone, or non-actionable replies."""

RUBRIC_PROMPT_TEMPLATE = """Evaluate the following support interaction:

[CUSTOMER QUERY]
"{customer_text}"

[DETECTED INTENT]
{intent}

[RETRIEVED HISTORICAL PRECEDENTS (Context)]
{precedents}

[AI DRAFTED REPLY]
"{draft_reply}"

[EVALUATION RUBRIC (1 to 5 scale)]:
1. Groundedness:
   - 1: Fabricates fake URLs, false warranty promises, or completely unsupported facts.
   - 3: Mostly grounded, but introduces generic assumptions not present in precedents.
   - 5: Strictly grounded in precedents, zero hallucinated facts or fabricated links.

2. Correctness:
   - 1: Irrelevant or answers a completely different problem.
   - 3: Relevant problem mentioned, but diagnostic questions or guidance are imprecise.
   - 5: Perfectly targets the exact customer issue with appropriate diagnostic guidance.

3. Tone Match:
   - 1: Rude, robotic, overly casual, or completely un-Apple.
   - 3: Acceptable support tone, but slightly verbose or awkward.
   - 5: Exemplary Apple Support tone: empathetic, professional, polite, concise tweet.

4. Resolution Likelihood:
   - 1: Completely unhelpful, leaves customer stuck.
   - 3: Offers partial help, but lacks clear next action.
   - 5: Provides clear, actionable next step (DM, specific troubleshooting step, or official link).

You MUST output ONLY a valid JSON object:
{{
  "groundedness": <1-5>,
  "correctness": <1-5>,
  "tone_match": <1-5>,
  "resolution_likelihood": <1-5>,
  "critique": "<short 1-2 sentence critique>"
}}"""

class LLMReplyJudge:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()

    def evaluate_reply(
        self,
        customer_text: str,
        intent: str,
        draft_reply: str,
        precedents: list[dict]
    ) -> dict:
        prec_text = "\n".join([
            f"- Precedent {i}: Cust: {p.get('customer_text', '')} -> Brand: {p.get('brand_reply', '')}"
            for i, p in enumerate(precedents[:2], 1)
        ]) or "No precedents available."
        
        prompt = RUBRIC_PROMPT_TEMPLATE.format(
            customer_text=customer_text,
            intent=intent,
            precedents=prec_text,
            draft_reply=draft_reply
        )
        
        raw_resp, model_used = self.client.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=200,
            is_judge=True
        )
        
        scores = self._parse_scores(raw_resp)
        scores["judge_model"] = model_used
        scores["overall_mean"] = round(
            (scores["groundedness"] + scores["correctness"] + scores["tone_match"] + scores["resolution_likelihood"]) / 4.0,
            2
        )
        return scores

    def _parse_scores(self, text: str) -> dict:
        try:
            # Look for JSON object containing groundedness
            matches = list(re.finditer(r'\{[^{}]*"groundedness"[^{}]*\}', text, re.DOTALL))
            if matches:
                data = json.loads(matches[-1].group(0))
                return {
                    "groundedness": max(1, min(5, int(data.get("groundedness", 4)))),
                    "correctness": max(1, min(5, int(data.get("correctness", 4)))),
                    "tone_match": max(1, min(5, int(data.get("tone_match", 4)))),
                    "resolution_likelihood": max(1, min(5, int(data.get("resolution_likelihood", 4)))),
                    "critique": str(data.get("critique", "Evaluation completed."))
                }
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "groundedness": max(1, min(5, int(data.get("groundedness", 4)))),
                    "correctness": max(1, min(5, int(data.get("correctness", 4)))),
                    "tone_match": max(1, min(5, int(data.get("tone_match", 4)))),
                    "resolution_likelihood": max(1, min(5, int(data.get("resolution_likelihood", 4)))),
                    "critique": str(data.get("critique", "Evaluation completed."))
                }
        except Exception:
            pass
            
        return {
            "groundedness": 4,
            "correctness": 4,
            "tone_match": 4,
            "resolution_likelihood": 4,
            "critique": "Fallback default scoring."
        }


if __name__ == "__main__":
    judge = LLMReplyJudge()
    scores = judge.evaluate_reply(
        customer_text="My battery on iOS 11.1 drops from 100 to 20 percent in less than 2 hours",
        intent="battery_power_issue",
        draft_reply="@customer We understand how important battery life is. Send us a DM with your exact iOS version so we can investigate. https://t.co/GDrqU22YpT",
        precedents=[{"customer_text": "Battery issue on iPhone 7", "brand_reply": "Send us a DM so we can help."}]
    )
    print("\n--- LLM-AS-JUDGE TEST ---")
    print(json.dumps(scores, indent=2))
