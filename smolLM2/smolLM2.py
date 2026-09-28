import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and reference model...")

tok = AutoTokenizer.from_pretrained(MODEL)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
).to(device)

sd = hf_model.state_dict()
config = hf_model.config

hidden = config.hidden_size
layers = config.num_hidden_layers
heads = config.num_attention_heads
kv_heads = config.num_key_value_heads
head_dim = config.head_dim

repeat_factor = heads // kv_heads


def rope(x):
    # x: [batch, heads, sequence, head_dim]
    d = x.shape[-1]

    freqs = 10000.0 ** (
        -torch.arange(
            0,
            d,
            2,
            device=x.device,
            dtype=torch.float32,
        )
        / d
    )

    positions = torch.arange(
        x.shape[-2],
        device=x.device,
        dtype=torch.float32,
    )

    angles = positions[:, None] * freqs[None, :]

    cos = angles.cos()[None, None, :, :].to(x.dtype)
    sin = angles.sin()[None, None, :, :].to(x.dtype)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    out = torch.stack(
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )

    return out.flatten(-2)


def forward(input_ids):
    x = F.embedding(
        input_ids,
        sd["model.embed_tokens.weight"],
    )

    batch, seq_len, _ = x.shape

    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=x.device,
            dtype=torch.bool,
        )
    )[None, None, :, :]

    for i in range(layers):
        layer = f"model.layers.{i}"

        # Attention input normalization
        residual = x

        x = F.rms_norm(
            x,
            (hidden,),
            weight=sd[f"{layer}.input_layernorm.weight"],
            eps=1e-5,
        )

        # Query
        q = F.linear(
            x,
            sd[f"{layer}.self_attn.q_proj.weight"],
        )

        q = q.view(
            batch,
            seq_len,
            heads,
            head_dim,
        ).transpose(1, 2)

        q = rope(q)

        # Key
        k = F.linear(
            x,
            sd[f"{layer}.self_attn.k_proj.weight"],
        )

        k = k.view(
            batch,
            seq_len,
            kv_heads,
            head_dim,
        ).transpose(1, 2)

        k = rope(k)

        # Value
        v = F.linear(
            x,
            sd[f"{layer}.self_attn.v_proj.weight"],
        )

        v = v.view(
            batch,
            seq_len,
            kv_heads,
            head_dim,
        ).transpose(1, 2)

        # Grouped-query attention
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attention = weights @ v

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch, seq_len, hidden)

        attention = F.linear(
            attention,
            sd[f"{layer}.self_attn.o_proj.weight"],
        )

        x = residual + attention

        # MLP
        residual = x

        x = F.rms_norm(
            x,
            (hidden,),
            weight=sd[f"{layer}.post_attention_layernorm.weight"],
            eps=1e-5,
        )

        gate = F.linear(
            x,
            sd[f"{layer}.mlp.gate_proj.weight"],
        )

        up = F.linear(
            x,
            sd[f"{layer}.mlp.up_proj.weight"],
        )

        down = F.linear(
            F.silu(gate) * up,
            sd[f"{layer}.mlp.down_proj.weight"],
        )

        x = residual + down

    # Final normalization
    x = F.rms_norm(
        x,
        (hidden,),
        weight=sd["model.norm.weight"],
        eps=1e-5,
    )

    # Tied embedding/language-model head
    logits = F.linear(
        x,
        sd["model.embed_tokens.weight"],
    )

    return logits


prompt = "hello how are you?"
input_ids = tok(
    prompt,
    return_tensors="pt",
)["input_ids"].to(device)

generated_ids = input_ids.clone()

max_new_tokens = 30

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = forward(generated_ids)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1,
        )

        if next_token.item() == tok.eos_token_id:
            break


output = tok.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

print()
print("Prompt:", prompt)
print("Output:", output)
