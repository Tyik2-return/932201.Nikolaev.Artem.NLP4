from sys import argv
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

np.random.seed(42)
torch.manual_seed(42)

def generate(
            model, tok, text,
            do_sample=True, max_length=100, repetition_penalty=5.0,
            top_k=5, top_p=0.95, temperature=1,
            num_beams=None,
            no_repeat_ngram_size=3
            ):
          input_ids = tok.encode(text, return_tensors="pt")
          out = model.generate(
              input_ids,
              max_length=max_length,
              repetition_penalty=repetition_penalty,
              do_sample=do_sample,
              top_k=top_k, top_p=top_p, temperature=temperature,
              num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
              pad_token_id=tok.eos_token_id
              )
          return list(map(tok.decode, out))

def load_tokenizer_and_model(model_name_or_path):
    return GPT2Tokenizer.from_pretrained(model_name_or_path), GPT2LMHeadModel.from_pretrained(model_name_or_path)

model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
tok, model = load_tokenizer_and_model(model_name)

prompt = """"Тише!" - прошептал рыбак. "Любой шум заставит его убежать".
Я начал медленно разворачиваться, чувствуя, как холодеет от напряжения. Пытаясь осторожнее удить рыбу, чтобы не"""

generated = generate(
    model, tok, prompt,
    do_sample=False,
    max_length=120,
    repetition_penalty=6.0,
    top_k=3,
    top_p=0.9,
    temperature=0.2,
    num_beams=15,
    no_repeat_ngram_size=4
)

print(generated[0])

