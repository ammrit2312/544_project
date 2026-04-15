"""
Inference for GECModel trained with train_gec_ft.py (unfrozen encoder checkpoint).
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from model.gec_model import GECModel

CHECKPOINT = 'checkpoints/gec_model_encoder_ft.pt'
MAX_LEN    = 128

# Defaults match train_gec_ft.py; overridden by checkpoint['config'] if present
_DEFAULT_CFG = {
    'freeze_encoder': False,
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 4,
    'max_len': 128,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


device = get_device()
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

try:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(CHECKPOINT, map_location=device)

merged = {**_DEFAULT_CFG, **ckpt.get('config', {})}
max_len = int(merged['max_len'])

model = GECModel(
    d_model = merged['d_model'],
    num_heads      = merged['num_heads'],
    d_ff           = merged['d_ff'],
    num_layers     = merged['num_layers'],
    max_len        = max_len,
    freeze_encoder = merged['freeze_encoder'],
).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()


def correct(sentence: str, max_new_tokens: int | None = None) -> str:
    cap = max_new_tokens if max_new_tokens is not None else max_len
    src = tokenizer(
        sentence,
        max_length     = max_len,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt',
    )
    input_ids      = src['input_ids'].to(device)
    attention_mask = src['attention_mask'].to(device)

    with torch.no_grad():
        encoder_out = model.encode(input_ids, attention_mask)

    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    generated = [bos_id]
    for _ in range(cap):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.decoder(tgt, encoder_out, tgt_mask=model.make_tgt_mask(tgt))
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if next_id == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == '__main__':
    examples = [
        "She go to school yesterday .",
        "He don't likes coffee .",
        "Ella van al mercado cada dia .",
    ]
    for sent in examples:
        print(f"Input    : {sent}")
        print(f"Corrected: {correct(sent)}")
        print()
