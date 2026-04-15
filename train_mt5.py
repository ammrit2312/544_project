"""
Fine-tune mT5 for GEC: full seq2seq on (incorrect -> correct) pairs.
Prefix: "gec: " + source (common convention for T5-style models).
"""

import os
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

TRAIN_CSV   = 'data/processed/train.csv'
VAL_CSV     = 'data/processed/val.csv'
CHECKPOINT  = 'checkpoints/mt5_gec.pt'
MODEL_NAME  = os.getenv('MT5_MODEL', 'google/mt5-small')
BATCH_SIZE  = int(os.getenv('MT5_BATCH', '16'))
EPOCHS      = int(os.getenv('MT5_EPOCHS', '3'))
LR          = float(os.getenv('MT5_LR', '3e-5'))
MAX_SRC     = 128
MAX_TGT     = 128
PREFIX      = 'gec: '


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class MT5GECDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_src: int, max_tgt: int):
        self.df = pd.read_csv(csv_path).dropna(subset=['incorrect', 'correct'])
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = PREFIX + str(row['incorrect'])
        tgt = str(row['correct'])

        enc = self.tok(
            src,
            max_length=self.max_src,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        lab = self.tok(
            text_target=tgt,
            max_length=self.max_tgt,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        labels = lab['input_ids'].squeeze().clone()
        labels[labels == self.pad_id] = -100

        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': labels,
        }


def collate(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }


def run_epoch(model, loader, optimizer, device, scheduler=None, train=True):
    model.train() if train else model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            total += loss.item()
    return total / max(1, len(loader))


def main():
    os.makedirs('checkpoints', exist_ok=True)
    device = get_device()
    print(f"Device: {device}  Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    train_ds = MT5GECDataset(TRAIN_CSV, tokenizer, MAX_SRC, MAX_TGT)
    val_ds = MT5GECDataset(VAL_CSV, tokenizer, MAX_SRC, MAX_TGT)
    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=nw, pin_memory=device.type == 'cuda', collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=nw, pin_memory=device.type == 'cuda', collate_fn=collate,
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, scheduler, train=True)
        va = run_epoch(model, val_loader, optimizer, device, scheduler=None, train=False)
        print(
            f"Epoch {epoch}/{EPOCHS} | Train loss: {tr:.4f} | Val loss: {va:.4f} | "
            f"{(time.time() - t0) / 60:.1f}m"
        )
        if va < best_val:
            best_val = va
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'val_loss': va,
                    'model_name': MODEL_NAME,
                },
                CHECKPOINT,
            )
            print(f"  Saved {CHECKPOINT}")
    print("Done.")


if __name__ == '__main__':
    main()
