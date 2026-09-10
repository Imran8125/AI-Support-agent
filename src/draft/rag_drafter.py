"""RAG Reply Drafter for AppleSupport.

Retrieves top-k (k=3) similar resolved precedents from Resolution KB using
cosine similarity on local BGE embeddings.
Drafts a grounded reply strictly adhering to historical brand tone and guidance.
Includes a hallucination/ungrounded claim checker.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import numpy as np
from src.utils.embeddings import LocalEmbeddingModel
from src.utils.llm_client import LLMClient

KB_JSON_PATH = "data/processed/resolution_kb.json"
KB_VECTORS_PATH = "data/processed/resolution_kb_vectors.npy"

SYSTEM_PROMPT = """You are an official customer support agent for Apple Support on Twitter (@AppleSupport).
Your replies must be polite, empathetic, concise, and technically accurate.
Do NOT invent serial numbers, order numbers, refund amounts, or false policies.
Ground your response strictly in the historical precedents provided.
Keep your response concise (under 240 characters) so it fits in a tweet."""

PROMPT_TEMPLATE = """Customer message: "{customer_text}"
Detected intent: {intent}

Here is how Apple Support has historically resolved similar issues:
{retrieved_precedents}

Draft an official Apple Support reply consistent with this tone and historical resolution pattern.
Do NOT fabricate links or promise refunds not mentioned in precedents.
Reply:"""

class RAGReplyDrafter:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()
        self.embedder = LocalEmbeddingModel()
        
        # Load Resolution KB
        with open(KB_JSON_PATH) as f:
            self.kb_items = json.load(f)
        self.kb_vectors = np.load(KB_VECTORS_PATH)
        
    def retrieve_similar(self, query: str, k: int = 3) -> list[dict]:
        """Dense cosine similarity retrieval over Resolution KB."""
        q_vec = self.embedder.embed_query(query)
        # Cosine similarity (both query and KB vectors are L2-normalized)
        sims = np.dot(self.kb_vectors, q_vec)
        top_indices = np.argsort(sims)[::-1][:k]
        
        results = []
        for idx in top_indices:
            item = dict(self.kb_items[idx])
            item["similarity"] = float(sims[idx])
            results.append(item)
        return results

    def draft_reply(self, customer_text: str, intent: str, k: int = 3) -> dict:
        precedents = self.retrieve_similar(customer_text, k=k)
        
        formatted_precedents = []
        for i, p in enumerate(precedents, 1):
            clean_brand = re.sub(r'@\d+', '@customer', p['brand_reply'])
            formatted_precedents.append(
                f"[Precedent {i}] (Similarity: {p['similarity']:.2f})\n"
                f"Customer: {p['customer_text']}\n"
                f"AppleSupport Reply: {clean_brand}"
            )
        precedents_str = "\n\n".join(formatted_precedents)
        
        prompt = PROMPT_TEMPLATE.format(
            customer_text=customer_text,
            intent=intent,
            retrieved_precedents=precedents_str
        )
        
        draft, model_used = self.client.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=150
        )
        
        # Clean up any quotes or prefixes
        draft = draft.strip('"\n ')
        
        # Hallucination & grounding check
        has_hallucinated_url = False
        urls_in_draft = re.findall(r'https?://\S+', draft)
        urls_in_context = re.findall(r'https?://\S+', precedents_str)
        for u in urls_in_draft:
            if u not in urls_in_context and "apple.co" not in u and "t.co" not in u:
                has_hallucinated_url = True
                
        return {
            "draft_reply": draft,
            "char_count": len(draft),
            "model_used": model_used,
            "top_similarity": precedents[0]["similarity"] if precedents else 0.0,
            "precedents": precedents,
            "hallucination_flag": has_hallucinated_url
        }


if __name__ == "__main__":
    drafter = RAGReplyDrafter()
    test_query = "My battery on iOS 11.1 drops from 100 to 20 percent in less than 2 hours"
    res = drafter.draft_reply(test_query, intent="battery_power_issue")
    print("\n--- TEST RAG DRAFT ---")
    print(f"Customer: {test_query}")
    print(f"Draft:    {res['draft_reply']}")
    print(f"Top Sim:  {res['top_similarity']:.3f}")
    print(f"Model:    {res['model_used']}")
