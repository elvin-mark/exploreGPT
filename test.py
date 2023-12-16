import os
import dotenv
from gpt import GPT2, GPTConfig
from utils import get_encoder
from tqdm import tqdm
import random
import torch

dotenv.load_dotenv()

conf = GPTConfig()
gpt = GPT2(conf)
gpt.load_model(os.environ["GPT2_MODEL_PATH"])
gpt.eval()
tokenizer = get_encoder(os.environ["GPT2_TOKENIZER_PATH"])

prompt = "In physics, string theory is a theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects called strings."
top_k = 5
predict_tokens = 40

inputs = tokenizer.encode(prompt)
with torch.no_grad():
    for _ in tqdm(range(predict_tokens), "generating"):
        logits = gpt(torch.tensor(inputs).view((1, -1)))
        next_id = random.choice(torch.argsort(
            logits)[0, -1, -top_k:])
        inputs.append(int(next_id))

print(tokenizer.decode(inputs[len(inputs) - predict_tokens:]))
