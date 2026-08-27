import os
from typing import List, Dict
from groq import Groq


class LegalGenerator:
    """Handles the Generation phase using Groq API."""

    def __init__(
        self,
        model_name: str = "allam-2-7b"
    ):
        self.model_name = model_name

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add your Groq API key."
            )

        self.client = Groq(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=512
        )

        return response.choices[0].message.content