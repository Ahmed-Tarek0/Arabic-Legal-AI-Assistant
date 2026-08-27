"""
Legal Answer Generator using Gemini AI.
Provides direct, accurate, natural Arabic legal answers for uploaded contracts with ultra-fast latency.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from config import DEFAULT_GEMINI_API_KEY, DEFAULT_GEMINI_MODEL


class LegalGenerator:
    """
    Direct Gemini Generator for Arabic Contract Q&A.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self._gemini_client = None

        if self.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[warn] Gemini init error: {e}")
                self._gemini_client = None

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response given chat messages."""
        if not self._gemini_client:
            return "⚠️ تعذر الاتصال بمحرك الذكاء الاصطناعي (Gemini)."

        system_instruction = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        user_content = "\n\n".join([m["content"] for m in messages if m["role"] == "user"])

        # High-quota, ultra-low latency models
        candidate_models = [self.model_name, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash"]
        for model in candidate_models:
            try:
                response = self._gemini_client.models.generate_content(
                    model=model,
                    contents=user_content,
                    config={"system_instruction": system_instruction, "temperature": 0.1},
                )
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue

        return "❌ تعذر توليد الإجابة. يرجى المحاولة مرة أخرى."

    def generate_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Generate streaming response with lowest time-to-first-token."""
        if not self._gemini_client:
            yield "⚠️ تعذر الاتصال بمحرك الذكاء الاصطناعي (Gemini)."
            return

        system_instruction = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        user_content = "\n\n".join([m["content"] for m in messages if m["role"] == "user"])

        candidate_models = [self.model_name, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash"]
        for model in candidate_models:
            try:
                response = self._gemini_client.models.generate_content_stream(
                    model=model,
                    contents=user_content,
                    config={"system_instruction": system_instruction, "temperature": 0.1},
                )
                emitted = False
                for chunk in response:
                    if chunk.text:
                        emitted = True
                        yield chunk.text
                if emitted:
                    return
            except Exception:
                continue

        # If stream fails, fallback to generate
        yield self.generate(messages)