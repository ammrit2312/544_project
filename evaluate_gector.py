import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from gec_metrics import compute_exact_match, compute_f05
from infer_gector import correct

def evaluate(test_csv: str, sample: int=None, lang: str=None):
    df = pd.read_csv(test_csv).dropna(subset=['incorrect', 'correct'])
    if lang:
        df = df[df['lang'] == lang]
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    print(f'Evaluating {len(df)} sentences' + (f' [{lang}]' if lang else '') + '...')
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
    print('=== English ===')
    evaluate('data/processed/test.csv', sample=200, lang='en')
    print('\n=== Spanish ===')
    evaluate('data/processed/test.csv', sample=200, lang='es')
