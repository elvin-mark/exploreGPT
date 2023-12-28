import torch
import json
from functools import lru_cache
import regex as re


def load_encoder_hparams_and_params(model_path, tokenizer_path, device="cpu"):
    n_layers = 12
    # weights can be donwloaded from https://huggingface.co/gpt2 (pytorch_model.bin)
    model = torch.load(model_path, map_location=device)
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 768]
    wte = model["wte.weight"].numpy()
    # context embedding. shape [ctx_len, emb_dim] ex. [1024, 768]
    wpe = model["wpe.weight"].numpy()

    # transformer blocks
    blocks = []
    for i in range(n_layers):
        # MultiHead attention weights
        c_attn = {"w": model[f"h.{i}.attn.c_attn.weight"].numpy(),
                  "b": model[f"h.{i}.attn.c_attn.bias"].numpy()}
        c_proj = {"w": model[f"h.{i}.attn.c_proj.weight"].numpy(),
                  "b": model[f"h.{i}.attn.c_proj.bias"].numpy()}
        attn = {"c_attn": c_attn, "c_proj": c_proj}

        # Layer Normalization weights
        ln_1 = {"g": model[f"h.{i}.ln_1.weight"].numpy(),
                "b": model[f"h.{i}.ln_1.bias"].numpy()}

        # MLP
        mlp_c_fc = {"w": model[f"h.{i}.mlp.c_fc.weight"].numpy(),
                    "b": model[f"h.{i}.mlp.c_fc.bias"].numpy()}
        mlp_c_proj = {"w": model[f"h.{i}.mlp.c_proj.weight"].numpy(),
                      "b": model[f"h.{i}.mlp.c_proj.bias"].numpy()}
        mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

        # Layer Normalization weights
        ln_2 = {"g": model[f"h.{i}.ln_2.weight"].numpy(),
                "b": model[f"h.{i}.ln_2.bias"].numpy()}
        block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
        blocks.append(block)
    # Last Layer Normalization weights
    ln_f = {"g": model["ln_f.weight"].numpy(), "b": model["ln_f.bias"].numpy()}
    params = {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f": ln_f
    }
    # Hyperparameters for the GTP2 124M model
    hparams = {}
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    return get_encoder(tokenizer_path), hparams, params


"""Byte pair encoding utilities.

Copied from: https://github.com/openai/gpt-2/blob/master/src/encoder.py.
"""


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"),
                                                          ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_data, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b]
                            for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c]
                         for c in text]).decode("utf-8", errors=self.errors)
        return text


def get_encoder(tokenizer_path):
    with open(tokenizer_path, "r",encoding="utf-8") as f:
        tokenizer = json.load(f)
    return Encoder(encoder=tokenizer["model"]["vocab"], bpe_data=tokenizer["model"]["merges"])
