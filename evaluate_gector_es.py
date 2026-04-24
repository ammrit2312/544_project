import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from infer_gector_es import correct

def _lcs_length(a, b):
    m, n = (len(a), len(b))
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]

def _norm_text(s: str) -> str:
    return ' '.join(str(s).lower().split())

def compute_exact_match(predictions, references):
    n = len(predictions)
    if n == 0:
        return 0.0
    hits = sum((_norm_text(p) == _norm_text(r) for p, r in zip(predictions, references)))
    return round(hits / n, 4)

def compute_f05(predictions, references):
    beta = 0.5
    total_lcs = total_pred = total_ref = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        lcs = _lcs_length(pred_tokens, ref_tokens)
        total_lcs += lcs
        total_pred += len(pred_tokens)
        total_ref += len(ref_tokens)
    precision = total_lcs / (total_pred + 1e-08)
    recall = total_lcs / (total_ref + 1e-08)
    f05 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-08)
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f0.5': round(f05, 4)}

def evaluate(test_csv: str, sample: int=None):
    df = pd.read_csv(test_csv).dropna(subset=['incorrect', 'correct'])
    df = df[df['lang'] == 'es']
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    print(f'Evaluating {len(df)} Spanish sentences (xlm-roberta-base_gector_es)...')
    preds = []
    for i, row in enumerate(df.itertuples()):
        preds.append(correct(str(row.incorrect)))
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(df)}')
    refs = df['correct'].tolist()
    scores = compute_f05(preds, refs)
    scores['exact_match'] = compute_exact_match(preds, refs)
    print(f'\nPrecision    : {scores['precision']}')
    print(f'Recall       : {scores['recall']}')
    print(f'F0.5         : {scores['f0.5']}')
    print(f'Exact match  : {scores['exact_match']}')
    return scores
if __name__ == '__main__':
    evaluate('data/processed/test.csv', sample=200)
