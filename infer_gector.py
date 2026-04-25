import os
import pickle
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from train_gector import GECTorModel, KEEP, DELETE
CHECKPOINT = 'checkpoints/gector_model.pt'
LABEL_VOCAB = 'checkpoints/gector_labels_word.pkl'
ENCODER = 'xlm-roberta-base'
MAX_LEN = 128
_DEFAULT_MAX_PASSES = max(1, int(os.getenv('GECTOR_MAX_PASSES', '3')))

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(ENCODER)
with open(LABEL_VOCAB, 'rb') as f:
    labels = pickle.load(f)
id2label = {i: lbl for i, lbl in enumerate(labels)}
try:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
except TypeError:
    ckpt = torch.load(CHECKPOINT, map_location=device)
model = GECTorModel(ENCODER, num_labels=len(labels)).to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

def _correct_once(sentence: str) -> str:
    enc = tokenizer(sentence, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    preds = logits[0].argmax(dim=-1).tolist()
    word_ids = enc.word_ids(batch_index=0)
    src_words = sentence.split()
    word_label: dict[int, str] = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in word_label:
            word_label[wid] = id2label.get(preds[pos], KEEP)
    output_tokens = []
    for wid, word in enumerate(src_words):
        lbl = word_label.get(wid, KEEP)
        if lbl == KEEP:
            output_tokens.append(word)
        elif lbl == DELETE:
            pass
        elif lbl.startswith('$REPLACE_'):
            replacement = lbl[len('$REPLACE_'):]
            output_tokens.append(replacement)
        elif lbl.startswith('$INSERT_'):
            insert_tok = lbl[len('$INSERT_'):]
            output_tokens.append(insert_tok)
            output_tokens.append(word)
        else:
            output_tokens.append(word)
    return ' '.join(output_tokens)

def correct(sentence: str, max_passes: int | None=None) -> str:
    n = _DEFAULT_MAX_PASSES if max_passes is None else max(1, max_passes)
    s = sentence
    for _ in range(n):
        nxt = _correct_once(s)
        if nxt == s:
            break
        s = nxt
    return s
if __name__ == '__main__':
    examples = ['She go to school yesterday .', "He don't likes coffee .", 'Ella van al mercado cada dia .']
    for sent in examples:
        print(f'Input    : {sent}')
        print(f'Corrected: {correct(sent)}')
        print(f'(1-pass) : {_correct_once(sent)}')
        print()
