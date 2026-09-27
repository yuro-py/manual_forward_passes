import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer and reference model...")
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct" # USE official model name from huggingface
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"

hf_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
sd = hf_model.state_dict()

layers = 32

# vocab = F.embedding(sd.embed_tokens, 49152, 960, padding_idx=2) # this has a bug I know that

vocab = torch.randn([960, 960])

def attn(q,k,v,o):
    scores = (q @ k.transpose) / (960 ** 0.5)
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    attention = weights @ v
    o = o(attention.transpose(-1,-2).replace(320, 960))

def ffn(x): # BOILERPLATE
    # gate_proj = nn.Linear(960, 2560, bias=False)
    # up_proj = 960, 2560, bias=False
    # down_proj = 2560, 960, bias=False)
    # activation = silu()
    pass

def rmsnorm(): # BOILERPLATE
    pass

def rope(): # BOILERPLATE
    pass

h = 12
hd = 960 // h
kv_h = 4
kv_hd = hd // 3

for i in range(layers):

    q = F.linear(vocab, sd['model.layers.i.self_attn.q_proj.weight'])
    q = q.reshape(960, h, hd).transpose(-1, -2)
    
    k = F.linear(vocab, sd['model.layers.i.self_attn.k_proj.weight'])
    k = k.reshape(320, kv_h, kv_hd).transpose(-1, -2)
    
    v = F.linear(vocab, sd['model.layers.i.self_attn.v_proj.weight'])
    v = v.reshape(320, kv_h, kv_hd).transpose(-1, -2)
    # I didnt do the broadcasting/interleaving of gqa's kv and q head.
    o = nn.Linear(960, 960, bias=False)

    attention = attn(q,k,v,o)

    mlp = ffn(attention)

    input_layernorm = rmsnorm(mlp) # wrong. how to use "model.layers.i.input_layernorm.weight"
    post_attention_layernorm = rmsnorm(input_layernorm) # wrong. how to use "model.layers.i.post_attention_layernorm.weight"

# how to use "(input_layernorm): LlamaRMSNorm((960,), eps=1e-05)" outside decoderlayer?

# how to use "(rotary_emb): LlamaRotaryEmbedding()" outside decoder layer?

# how to use this lm_head thing?what is 49152?
lm_head = nn.Linear(960, 49152, bias=False)
