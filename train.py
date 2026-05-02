import os
import time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import GECDataset
from model.gec_model import GECModel

TRAIN_CSV   = 'data/processed/train.csv'
VAL_CSV     = 'data/processed/val.csv'
CHECKPOINT  = 'checkpoints/gec_model.pt'
BATCH_SIZE  = 32
EPOCHS      = 25
LR          = 1e-4
MAX_LEN     = 128
D_MODEL     = 512
NUM_HEADS   = 8
D_FF        = 2048
NUM_LAYERS  = 4

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

device = get_device()
print(f"Device: {device}")

train_dataset = GECDataset(TRAIN_CSV, max_len=MAX_LEN)
val_dataset   = GECDataset(VAL_CSV,   max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)

model = GECModel(
    d_model    = D_MODEL,
    num_heads  = NUM_HEADS,
    d_ff       = D_FF,
    num_layers = NUM_LAYERS,
    max_len    = MAX_LEN
).to(device)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

optimizer = torch.optim.AdamW(trainable, lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps  = total_steps // 10,
    num_training_steps= total_steps
)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

os.makedirs('checkpoints', exist_ok=True)
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tgt_input      = batch['tgt_input'].to(device)
        tgt_output     = batch['tgt_output'].to(device)

        logits = model(input_ids, attention_mask, tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tgt_input      = batch['tgt_input'].to(device)
            tgt_output     = batch['tgt_output'].to(device)

            logits = model(input_ids, attention_mask, tgt_input)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)
    epoch_mins = (time.time() - epoch_start) / 60
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Time: {epoch_mins:.1f}m")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch'      : epoch + 1,
            'model_state': model.state_dict(),
            'val_loss'   : avg_val
        }, CHECKPOINT)
        print(f"Saved best model (epoch {epoch+1})")

print("\nTraining complete.")


