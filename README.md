# exploreGPT

This is a personal repo to explore how GPT works and how to train a simple GPT.

# Simple Inference

You will need to download the weights and the tokenizer vocabulary from https://huggingface.co/gpt2

```sh
python simple_inference.py --topk 5 "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."
```

```sh
python test.py "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."
```
