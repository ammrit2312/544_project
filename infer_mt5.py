"""mT5 GEC inference (must match train_mt5 prefix and checkpoint)."""

import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

CHECKPOINT = 'checkpoints/mt5_gec.pt'
MAX_LEN    = 128
PREFIX     = 'gec: '


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


device = get_device()
try:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(CHECKPOINT, map_location=device)
model_name = ckpt.get('model_name', 'google/mt5-small')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()


def correct(sentence: str, max_new_tokens: int = MAX_LEN) -> str:
    text = PREFIX + sentence
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        return_tensors='pt',
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == '__main__':
    for s in [
        "She go to school yesterday .",
        "He don't likes coffee .",
    ]:
        print("In:", s)
        print("Out:", correct(s))
        print()
