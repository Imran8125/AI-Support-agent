"""Unified LLM Client supporting OpenRouter (free-tier rotation) and local LM Studio (e.g. gemma-4-e4b).

Includes:
- SQLite response caching by SHA-256 prompt hash
- Rate-limit aware exponential backoff retries
- Dynamic model rotation across free-tier candidates
- Configurable via .env
"""

from __future__ import annotations

import os
import sqlite3
import hashlib
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, cache_db: str = "data/cache.sqlite"):
        self.provider = os.getenv("LLM_PROVIDER", "auto").lower()
        self.lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.lm_studio_model = os.getenv("LM_STUDIO_MODEL", "gemma-4-e4b")
        
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        models_str = os.getenv(
            "OPENROUTER_MODELS",
            "nex-agi/nex-n2.5-pro:free,nex-agi/nex-n2.5-mini:free,nvidia/nemotron-3-super-120b-a12b:free,openrouter/free"
        )
        self.openrouter_models = [m.strip() for m in models_str.split(",") if m.strip()]
        self.judge_model = os.getenv("OPENROUTER_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        self.model_rotation_idx = 0
        
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_db = cache_db
        self._init_cache()

    def _init_cache(self):
        os.makedirs(os.path.dirname(self.cache_db) or ".", exist_ok=True)
        with sqlite3.connect(self.cache_db, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    provider TEXT,
                    model TEXT,
                    system_prompt TEXT,
                    prompt TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _compute_hash(self, provider: str, model: str, system_prompt: str, prompt: str, temperature: float) -> str:
        content = f"{provider}|{model}|{system_prompt}|{prompt}|{temperature:.2f}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_from_cache(self, prompt_hash: str) -> str | None:
        if not self.cache_enabled:
            return None
        with sqlite3.connect(self.cache_db, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT response FROM prompt_cache WHERE prompt_hash = ?", (prompt_hash,))
            row = cursor.fetchone()
            if row:
                return row[0]
        return None

    def _save_to_cache(self, prompt_hash: str, provider: str, model: str, system_prompt: str, prompt: str, response: str):
        if not self.cache_enabled:
            return
        with sqlite3.connect(self.cache_db, timeout=30.0) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO prompt_cache (prompt_hash, provider, model, system_prompt, prompt, response)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (prompt_hash, provider, model, system_prompt, prompt, response))
            conn.commit()

    def _check_lm_studio_alive(self) -> bool:
        try:
            r = requests.get(f"{self.lm_studio_base_url}/models", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert customer support agent for Apple.",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 500,
        is_judge: bool = False
    ) -> tuple[str, str]:
        """Generate response and return (response_text, model_used)."""
        
        # Decide target provider and model
        chosen_provider = self.provider
        if chosen_provider == "auto":
            if self._check_lm_studio_alive():
                chosen_provider = "lmstudio"
            elif self.openrouter_api_key:
                chosen_provider = "openrouter"
            else:
                # Neither ready yet - will raise helpful error
                chosen_provider = "lmstudio"

        if chosen_provider == "lmstudio":
            target_model = model or self.lm_studio_model
        else:
            if is_judge:
                target_model = self.judge_model
            elif model:
                target_model = model
            else:
                # Rotate across free models
                target_model = self.openrouter_models[self.model_rotation_idx % len(self.openrouter_models)]
                self.model_rotation_idx += 1

        # Check cache
        p_hash = self._compute_hash(chosen_provider, target_model, system_prompt, prompt, temperature)
        cached = self._get_from_cache(p_hash)
        if cached is not None:
            return cached, f"{target_model} (cached)"

        # Perform actual call
        if chosen_provider == "lmstudio":
            res = self._call_openai_compatible(
                base_url=self.lm_studio_base_url,
                api_key="not-needed",
                model=target_model,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            if not self.openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is not set in .env and LM Studio is not reachable on localhost:1234. "
                    "Please either start LM Studio or add your OpenRouter API key to .env."
                )
            res = self._call_openrouter(
                model=target_model,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        self._save_to_cache(p_hash, chosen_provider, target_model, system_prompt, prompt, res)
        return res, target_model

    def _call_openai_compatible(self, base_url: str, api_key: str, model: str, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure reasoning models (like gemma-4-e4b) have enough headroom for thinking tokens + completion
        is_judge_eval = "groundedness" in prompt.lower() or "rubric" in prompt.lower() or "judge" in system_prompt.lower()
        effective_max_tokens = max(max_tokens, 1200 if is_judge_eval else 600)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": effective_max_tokens
        }

        r = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"LM Studio API returned error {r.status_code}: {r.text}")
        data = r.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if not content.strip():
            content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return content.strip()

    def _call_openrouter(self, model: str, system_prompt: str, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/hiver-take-home/apple-support-agent",
            "X-Title": "AppleSupport AI Agent",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Handle rate limits with resilient retries
        max_retries = 6
        for attempt in range(max_retries):
            try:
                r = requests.post(f"{self.openrouter_base_url}/chat/completions", headers=headers, json=payload, timeout=60)
            except Exception as net_err:
                print(f"Network error on attempt {attempt+1}: {net_err}. Retrying...")
                time.sleep(3)
                continue

            if r.status_code == 200:
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content") or msg.get("reasoning") or ""
                if content.strip():
                    return content.strip()
                # Empty content - retry with rotation
            elif r.status_code in [429, 502, 503]:
                wait_time = (attempt + 1) * 8  # 8s, 16s, 24s...
                print(f"OpenRouter rate-limited ({r.status_code}) on {model}. Backing off {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"OpenRouter status {r.status_code} on {model}: {r.text[:80]}. Rotating...")
                time.sleep(2)

            # Rotate to next model in pool
            self.model_rotation_idx += 1
            model = self.openrouter_models[self.model_rotation_idx % len(self.openrouter_models)]
            payload["model"] = model

        # If all retries exhausted, return a safe fallback message
        print(f"Warning: Exhausted {max_retries} attempts on OpenRouter for prompt.")
        return json.dumps({"intent": "other_unclear", "confidence": 0.4, "reasoning": "Fallback on rate limit."})


if __name__ == "__main__":
    client = LLMClient()
    print(f"Active Provider mode: {client.provider}")
    print(f"LM Studio reachable: {client._check_lm_studio_alive()}")
    print(f"OpenRouter key present: {bool(client.openrouter_api_key)}")
