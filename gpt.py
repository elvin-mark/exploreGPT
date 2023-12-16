import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTConfig:

    def __init__(self, n_ctx=1024, n_heads=12, vocab_size=50257, emb_dim=768, n_layers=12):
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers


class Attention(nn.Module):
    def __init__(self, conf: GPTConfig):
        super(Attention, self).__init__()
        self.conf = conf
        self.c_attn = nn.Linear(conf.emb_dim, 3 * conf.emb_dim)
        self.c_proj = nn.Linear(conf.emb_dim, conf.emb_dim)
        self.register_buffer("bias", torch.tril(torch.ones(conf.n_ctx, conf.n_ctx))
                             .view(1, 1, conf.n_ctx, conf.n_ctx))

    def forward(self, x):
        # B: Batch size, T: number of tokens or sequence length, C: embedded dimension
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.conf.emb_dim, dim=2)
        k = k.view(B, T, self.conf.n_heads, C //
                   self.conf.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.conf.n_heads, C //
                   self.conf.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.conf.n_heads, C //
                   self.conf.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, conf: GPTConfig):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(conf.emb_dim, 3 * conf.n_ctx)
        self.c_proj = nn.Linear(3 * conf.n_ctx, conf.emb_dim)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, conf: GPTConfig):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(conf.emb_dim)
        self.ln_2 = nn.LayerNorm(conf.emb_dim)
        self.attn = Attention(conf)
        self.mlp = MLP(conf)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, conf=GPTConfig()):
        super(GPT2, self).__init__()
        self.conf = conf
        self.wte = nn.Embedding(conf.vocab_size, conf.emb_dim)
        self.wpe = nn.Embedding(conf.n_ctx, conf.emb_dim)
        self.blocks = []
        for _ in range(conf.n_layers):
            self.blocks.append(Block(conf))
        self.ln_f = nn.LayerNorm(conf.emb_dim)

    def forward(self, x, target=None):
        # B: Batch size, T: number of tokens or sequence length
        B, T = x.size()
        x = self.wte(x) + self.wpe(torch.arange(0, T, dtype=int))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T

    def load_model(self, model_path: str):
        model = torch.load(model_path, map_location="cpu")
        self.wte.weight.data = model["wte.weight"]
        self.wpe.weight.data = model["wpe.weight"]
        for i, block in enumerate(self.blocks):
            block.ln_1.weight.data = model[f"h.{i}.ln_1.weight"]
            block.ln_1.bias.data = model[f"h.{i}.ln_1.bias"]
            block.attn.c_attn.weight.data = model[f"h.{i}.attn.c_attn.weight"].T
            block.attn.c_attn.bias.data = model[f"h.{i}.attn.c_attn.bias"]
            block.attn.c_proj.weight.data = model[f"h.{i}.attn.c_proj.weight"].T
            block.attn.c_proj.bias.data = model[f"h.{i}.attn.c_proj.bias"]
            block.ln_2.weight.data = model[f"h.{i}.ln_2.weight"]
            block.ln_2.bias.data = model[f"h.{i}.ln_2.bias"]
            block.mlp.c_fc.weight.data = model[f"h.{i}.mlp.c_fc.weight"].T
            block.mlp.c_fc.bias.data = model[f"h.{i}.mlp.c_fc.bias"]
            block.mlp.c_proj.weight.data = model[f"h.{i}.mlp.c_proj.weight"].T
            block.mlp.c_proj.bias.data = model[f"h.{i}.mlp.c_proj.bias"]
        self.ln_f.weight.data = model["ln_f.weight"]
        self.ln_f.bias.data = model["ln_f.bias"]
