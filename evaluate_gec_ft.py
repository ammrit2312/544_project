"""
Evaluate GECModel fine-tuned with train_gec_ft.py (uses infer_gec_ft.correct).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from infer_gec_ft import correct


def compute_f05(predictions, references):
    """Token-level F0.5 (precision-weighted), same as evaluate.py."""
    beta = 0.5
    tp = fp = fn = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens  = set(ref.lower().split())
        tp += len(pred_tokens & ref_tokens)
        fp += len(pred_tokens - ref_tokens)
        fn += len(ref_tokens  - pred_tokens)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

    return {
        'precision': round(precision, 4),
        'recall'   : round(recall,    4),
        'f0.5'     : round(f05,       4),
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
        pred = correct(str(row.incorrect))
        preds.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")

    scores = compute_f05(preds, df['correct'].tolist())

    print(f"\nPrecision : {scores['precision']}")
    print(f"Recall    : {scores['recall']}")
    print(f"F0.5      : {scores['f0.5']}")
    return scores


if __name__ == '__main__':
    print("=== English (gec_model_encoder_ft) ===")
    evaluate('data/processed/test.csv', sample=200, lang='en')

    print("\n=== Spanish (gec_model_encoder_ft) ===")
    evaluate('data/processed/test.csv', sample=200, lang='es')
