# llm.py
from huggingface_hub import InferenceClient

import os
HF_TOKEN = os.getenv("HF_TOKEN")


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# Or Mistral:
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(
    MODEL_ID,
    token=HF_TOKEN
)

def run_llm(prompt: str) -> str:
    """Call HF Inference API for chat models."""
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message["content"]
