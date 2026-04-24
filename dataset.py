import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class GECDataset(Dataset):

    def __init__(self, csv_path, max_len=128):
        self.df = pd.read_csv(csv_path).dropna(subset=['incorrect', 'correct'])
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = str(row['incorrect'])
        tgt = str(row['correct'])
        src_enc = self.tokenizer(src, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_ids = tgt_enc['input_ids'].squeeze()
        tgt_input = tgt_ids[:-1]
        tgt_output = tgt_ids[1:]
        tgt_output = tgt_output.clone()
        tgt_output[tgt_output == self.tokenizer.pad_token_id] = -100
        return {'input_ids': src_enc['input_ids'].squeeze(), 'attention_mask': src_enc['attention_mask'].squeeze(), 'tgt_input': tgt_input, 'tgt_output': tgt_output}
