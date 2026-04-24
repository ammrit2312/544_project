import os
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from dataset2 import GECDataset, GECDataCollator
from model2 import GECModel

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print('Using device: mps')
        return torch.device('mps')
    if torch.cuda.is_available():
        print('Using device: cuda')
        return torch.device('cuda')
    print('Using device: cpu')
    return torch.device('cpu')

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, scheduler=None) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc='train', leave=False)
    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        source_ids = batch['source_ids']
        source_mask = batch['source_mask']
        target_input_ids = batch['target_input_ids']
        target_labels = batch['target_labels']
        optimizer.zero_grad(set_to_none=True)
        logits = model(source_ids=source_ids, target_ids=target_input_ids, source_mask=source_mask)
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), target_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, desc='val', leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        logits = model(source_ids=batch['source_ids'], target_ids=batch['target_input_ids'], source_mask=batch['source_mask'])
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), batch['target_labels'].view(-1))
        total_loss += loss.item()
        progress.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(1, len(loader))

@torch.no_grad()
def sample_generate(model: GECModel, tokenizer, device: torch.device, text: str, source_lang: Optional[str]=None, max_new_tokens: int=48):
    model.eval()
    if source_lang is not None:
        text = f'<{source_lang}> {text}'
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    source_ids = enc['input_ids'].to(device)
    source_mask = enc['attention_mask'].to(device)
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id
    generated = model.generate(source_ids=source_ids, source_mask=source_mask, bos_id=bos_id, eos_id=eos_id, max_new_tokens=max_new_tokens)
    print('INPUT:', text)
    print('OUTPUT:', tokenizer.decode(generated[0], skip_special_tokens=True))

def main():
    train_csv = 'data/processed/train.csv'
    val_csv = 'data/processed/val.csv'
    out_dir = Path('checkpoints')
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_name = 'xlm-roberta-base'
    batch_size = 64
    encoder_lr = 2e-05
    decoder_lr = 0.0003
    epochs = 30
    max_source_len = 128
    max_target_len = 128
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    freeze_encoder = False
    train_subset_size = None
    val_subset_size = None
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    train_ds = GECDataset(csv_path=train_csv, tokenizer=tokenizer, max_source_len=max_source_len, max_target_len=max_target_len, add_lang_token=True)
    val_ds = GECDataset(csv_path=val_csv, tokenizer=tokenizer, max_source_len=max_source_len, max_target_len=max_target_len, add_lang_token=True)
    if train_subset_size is not None and len(train_ds) > train_subset_size:
        train_ds.df = train_ds.df.sample(train_subset_size, random_state=42).reset_index(drop=True)
        print(f'Using train subset: {len(train_ds)} rows')
    else:
        print(f'Using full train set: {len(train_ds)} rows')
    if val_subset_size is not None and len(val_ds) > val_subset_size:
        val_ds.df = val_ds.df.sample(val_subset_size, random_state=42).reset_index(drop=True)
        print(f'Using val subset: {len(val_ds)} rows')
    else:
        print(f'Using full val set: {len(val_ds)} rows')
    collator = GECDataCollator()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True)
    print('train batches per epoch:', len(train_loader))
    print('total epochs:', epochs)
    print('total training steps:', len(train_loader) * epochs)
    model = GECModel(encoder_name=encoder_name, vocab_size=tokenizer.vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=d_ff, pad_idx=tokenizer.pad_token_id, freeze_encoder=freeze_encoder, max_len=max(max_source_len, max_target_len), dropout=0.1).to(device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in list(model.proj.parameters()) + list(model.decoder.parameters()) if p.requires_grad]
    trainable_params = encoder_params + decoder_params
    optimizer = torch.optim.AdamW([{'params': encoder_params, 'lr': encoder_lr}, {'params': decoder_params, 'lr': decoder_lr}])
    total_steps = len(train_loader) * epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print(f'Trainable parameters: {sum((p.numel() for p in trainable_params)):,}')
    print(f'Frozen encoder: {freeze_encoder}')
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'  train_loss = {train_loss:.4f}')
        print(f'  val_loss   = {val_loss:.4f}')
        ckpt_path = out_dir / f'epoch_{epoch}.pt'
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': train_loss, 'val_loss': val_loss, 'config': {'encoder_name': encoder_name, 'd_model': d_model, 'num_heads': num_heads, 'num_layers': num_layers, 'd_ff': d_ff, 'freeze_encoder': freeze_encoder, 'batch_size': batch_size, 'max_source_len': max_source_len, 'max_target_len': max_target_len, 'train_subset_size': train_subset_size, 'val_subset_size': val_subset_size}}, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            print('  saved best_model.pt')
    print('\nSanity check generation:')
    sample_generate(model, tokenizer, device, 'I has a book', source_lang='en')
    sample_generate(model, tokenizer, device, 'Yo tiene un libro', source_lang='es')
if __name__ == '__main__':
    main()
