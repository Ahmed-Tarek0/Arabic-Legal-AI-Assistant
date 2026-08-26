import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LegalGenerator:
    """Handles the Generation phase using Qwen 2.5 with 4-bit Quantization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]