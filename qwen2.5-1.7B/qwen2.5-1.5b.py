import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and reference model...")

tok = AutoTokenizer.from_pretrained(MODEL)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
).to(device)

sd = hf_model.state_dict()
config = hf_model.config
