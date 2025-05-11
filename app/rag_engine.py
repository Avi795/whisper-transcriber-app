# app/rag_engine.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base", torch_dtype=torch.float16, device_map="auto")

def generate_answer(query, contexts):
    prompt = """You are an AI assistant. Answer the question based on the following context:

"""
    for i, ctx in enumerate(contexts):
        prompt += f"Chunk {i+1}: {ctx}\n"
    prompt += f"\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

