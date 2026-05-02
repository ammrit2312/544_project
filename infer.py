import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from model.gec_model import GECModel

CHECKPOINT = 'checkpoints/gec_model.pt'
MAX_LEN    = 128

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device    = get_device()
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

model = GECModel().to(device)
ckpt  = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

def correct(sentence: str, rep_penalty: float = 1.5) -> str:
    src = tokenizer(
        sentence,
        max_length     = MAX_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )
    input_ids      = src['input_ids'].to(device)
    attention_mask = src['attention_mask'].to(device)

    src_len = int(attention_mask.sum().item())
    max_new_tokens = min(src_len + 10, MAX_LEN)

    with torch.no_grad():
        encoder_out = model.encode(input_ids, attention_mask)  # [1, src_len, d_model]

    src_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    num_beams = 4

    beams = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_new_tokens):
        candidates = []
        for score, seq in beams:
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model.decoder(
                    tgt, encoder_out,
                    src_mask=src_mask,
                    tgt_mask=model.make_tgt_mask(tgt)
                )
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            if rep_penalty != 1.0:
                for tok in set(seq):
                    if log_probs[tok] < 0:
                        log_probs[tok] *= rep_penalty
                    else:
                        log_probs[tok] /= rep_penalty

            topk = log_probs.topk(num_beams)
            for log_p, tok_id in zip(topk.values, topk.indices):
                candidates.append((score + log_p.item(), seq + [tok_id.item()]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for s, seq in candidates[:num_beams]:
            if seq[-1] == eos_id:
                completed.append((s / len(seq), seq))  # length-normalize
            else:
                beams.append((s, seq))
        if not beams:
            break

    if completed:
        best = max(completed, key=lambda x: x[0])[1]
    else:
        best = max(beams, key=lambda x: x[0])[1]

    return tokenizer.decode(best, skip_special_tokens=True)


if __name__ == '__main__':
    examples = [
        "She go to school yesterday .",
        "He don't likes coffee .",
        "Ella van al mercado cada dia .",   # Spanish
    ]
    for sent in examples:
        print(f"Input    : {sent}")
        print(f"Corrected: {correct(sent)}")
        print()


