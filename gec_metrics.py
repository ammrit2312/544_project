def lcs_length(a, b):
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

def norm_text(s: str) -> str:
    return ' '.join(str(s).lower().split())

def compute_exact_match(predictions, references):
    n = len(predictions)
    if n == 0:
        return 0.0
    hits = sum((norm_text(p) == norm_text(r) for p, r in zip(predictions, references)))
    return round(hits / n, 4)

def compute_f05(predictions, references):
    beta = 0.5
    total_lcs = total_pred = total_ref = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        lcs = lcs_length(pred_tokens, ref_tokens)
        total_lcs += lcs
        total_pred += len(pred_tokens)
        total_ref += len(ref_tokens)
    precision = total_lcs / (total_pred + 1e-08)
    recall = total_lcs / (total_ref + 1e-08)
    f05 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-08)
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f0.5': round(f05, 4)}
