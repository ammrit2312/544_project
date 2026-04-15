import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from infer_mt5 import correct


def _lcs_length(a, b):
    m, n = len(a), len(b)
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
    precision = total_lcs / (total_pred + 1e-8)
    recall = total_lcs / (total_ref + 1e-8)
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f0.5': round(f05, 4),
    }


def evaluate(test_csv: str, sample: int = None, lang: str = None):
    df = pd.read_csv(test_csv).dropna(subset=['incorrect', 'correct'])
    if lang:
        df = df[df['lang'] == lang]
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    print(f"Evaluating {len(df)} sentences" + (f" [{lang}]" if lang else "") + "...")
    preds = []
    for i, row in enumerate(df.itertuples()):
        preds.append(correct(str(row.incorrect)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")
    scores = compute_f05(preds, df['correct'].tolist())
    print(f"\nPrecision : {scores['precision']}")
    print(f"Recall    : {scores['recall']}")
    print(f"F0.5      : {scores['f0.5']}")
    return scores


if __name__ == '__main__':
    print("=== English ===")
    evaluate('data/processed/test.csv', sample=200, lang='en')
    print("\n=== Spanish ===")
    evaluate('data/processed/test.csv', sample=200, lang='es')
