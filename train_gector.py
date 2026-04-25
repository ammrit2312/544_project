import os
import time
import pickle
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
TRAIN_CSV = 'data/processed/train.csv'
VAL_CSV = 'data/processed/val.csv'
CHECKPOINT = 'checkpoints/gector_model.pt'
LABEL_VOCAB = 'checkpoints/gector_labels_word.pkl'
ENCODER_NAME = 'xlm-roberta-base'
BATCH_SIZE = 32
EPOCHS = int(os.getenv('GECTOR_EPOCHS', '20'))
LR = 2e-05
MAX_LEN = 128
TOP_K_LABELS = 5000
KEEP = '$KEEP'
DELETE = '$DELETE'

def build_label_vocab(csv_path: str, top_k: int, lang: str | None=None) -> list[str]:
    df = pd.read_csv(csv_path).dropna(subset=['incorrect', 'correct'])
    if lang is not None:
        if 'lang' not in df.columns:
            raise ValueError("CSV has no 'lang' column; cannot filter by lang=" + repr(lang))
        df = df[df['lang'] == lang]
    counter: Counter = Counter()
    for _, row in df.iterrows():
        src_words = str(row['incorrect']).split()
        tgt_words = str(row['correct']).split()
        labels = align_labels(src_words, tgt_words)
        counter.update(labels)
    fixed = [KEEP, DELETE]
    top = [lbl for lbl, _ in counter.most_common(top_k) if lbl not in fixed]
    return fixed + top

def align_labels(src_tokens: list[str], tgt_tokens: list[str]) -> list[str]:
    m, n = (len(src_tokens), len(tgt_tokens))
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_tokens[i - 1] == tgt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    alignment = []
    i, j = (m, n)
    while i > 0 and j > 0:
        if src_tokens[i - 1] == tgt_tokens[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    alignment.reverse()
    matched_src = {s for s, _ in alignment}
    matched_tgt = {t for _, t in alignment}
    insert_map: dict[int, list[str]] = {}
    prev_src = -1
    tgt_ptr = 0
    for src_i, tgt_i in alignment:
        while tgt_ptr < tgt_i:
            if tgt_ptr not in matched_tgt:
                insert_map.setdefault(src_i, []).append(f'$INSERT_{tgt_tokens[tgt_ptr]}')
            tgt_ptr += 1
        tgt_ptr = tgt_i + 1
    labels = []
    for src_i, src_tok in enumerate(src_tokens):
        if src_i in insert_map and insert_map[src_i]:
            labels.append(insert_map[src_i][0])
            continue
        if src_i in matched_src:
            labels.append(KEEP)
        else:
            tgt_i = next((t for s, t in alignment if s > src_i), None)
            if tgt_i is not None and tgt_i > 0:
                candidate = tgt_tokens[tgt_i - 1]
                labels.append(f'$REPLACE_{candidate}')
            else:
                labels.append(DELETE)
    return labels

class GECTorDataset(Dataset):

    def __init__(self, csv_path: str, tokenizer, label2id: dict, max_len: int=MAX_LEN, lang: str | None=None):
        self.df = pd.read_csv(csv_path).dropna(subset=['incorrect', 'correct'])
        if lang is not None:
            if 'lang' not in self.df.columns:
                raise ValueError("CSV has no 'lang' column; cannot filter by lang=" + repr(lang))
            self.df = self.df[self.df['lang'] == lang].reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.keep_id = label2id[KEEP]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row['incorrect'])
        tgt_text = str(row['correct'])
        src_words = src_text.split()
        tgt_words = tgt_text.split()
        raw_labels = align_labels(src_words, tgt_words)
        enc = self.tok(src_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        word_ids = enc.word_ids(batch_index=0)
        visible = [w for w in word_ids if w is not None]
        n_words_kept = max(visible) + 1 if visible else 0
        raw_labels = raw_labels[:n_words_kept]
        label_ids = [self.label2id.get(lbl, self.keep_id) for lbl in raw_labels]
        token_labels = torch.full((self.max_len,), -100, dtype=torch.long)
        prev_word = None
        for pos, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word and word_id < len(label_ids):
                token_labels[pos] = label_ids[word_id]
            prev_word = word_id
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': token_labels}

class GECTorModel(nn.Module):

    def __init__(self, encoder_name: str, num_labels: int, dropout: float=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        return self.classifier(hidden)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def run_epoch(model, loader, optimizer, criterion, device, scheduler=None, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

def main(checkpoint: str | None=None, label_vocab: str | None=None, encoder_name: str | None=None, lang: str | None=None):
    checkpoint = checkpoint or CHECKPOINT
    label_vocab = label_vocab or LABEL_VOCAB
    encoder_name = encoder_name or ENCODER_NAME
    os.makedirs(os.path.dirname(checkpoint) or '.', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    device = get_device()
    tag = f' [{lang}]' if lang else ''
    print(f'Device: {device}{tag}  Encoder: {encoder_name}')
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    if Path(label_vocab).exists():
        with open(label_vocab, 'rb') as f:
            labels = pickle.load(f)
        print(f'Loaded label vocab: {len(labels)} labels')
    else:
        print('Building label vocab from training data...')
        labels = build_label_vocab(TRAIN_CSV, TOP_K_LABELS, lang=lang)
        lv_dir = os.path.dirname(label_vocab)
        if lv_dir:
            os.makedirs(lv_dir, exist_ok=True)
        with open(label_vocab, 'wb') as f:
            pickle.dump(labels, f)
        print(f'Label vocab size: {len(labels)}')
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    num_labels = len(labels)
    train_ds = GECTorDataset(TRAIN_CSV, tokenizer, label2id, lang=lang)
    val_ds = GECTorDataset(VAL_CSV, tokenizer, label2id, lang=lang)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')
    nw = int(os.getenv('GECTOR_NUM_WORKERS', '4'))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin)
    model = GECTorModel(encoder_name, num_labels).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable params: {sum((p.numel() for p in trainable)):,}')
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, scheduler, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, scheduler=None, train=False)
        elapsed = (time.time() - t0) / 60
        print(f'Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}m')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_loss': val_loss, 'num_labels': num_labels, 'encoder_name': encoder_name, 'lang': lang}, checkpoint)
            print(f'  Saved best model (epoch {epoch})')
    print('\nTraining complete.')
if __name__ == '__main__':
    main()
