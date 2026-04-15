from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class GECBatch:
    source_ids: torch.Tensor
    source_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_labels: torch.Tensor
    lang: Optional[list] = None


class GECDataset(Dataset):
    """
    Dataset for multilingual grammar error correction.

    Expected CSV columns by default:
      - incorrect: noisy input sentence
      - correct: corrected target sentence
      - lang: language code such as 'en' or 'es' (optional but supported)

    This class tokenizes both source and target using the same tokenizer.
    It also prepares shifted decoder inputs and labels for teacher forcing.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 128,
        max_target_len: int = 128,
        source_col: str = "incorrect",
        target_col: str = "correct",
        lang_col: str = "lang",
        add_lang_token: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source_col = source_col
        self.target_col = target_col
        self.lang_col = lang_col if lang_col in self.df.columns else None
        self.add_lang_token = add_lang_token and self.lang_col is not None

        required = [self.source_col, self.target_col]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_path}")

        self.df = self.df.dropna(subset=required).reset_index(drop=True)
        self.df[self.source_col] = self.df[self.source_col].astype(str).str.strip()
        self.df[self.target_col] = self.df[self.target_col].astype(str).str.strip()
        self.df = self.df[(self.df[self.source_col] != "") & (self.df[self.target_col] != "")].reset_index(drop=True)

        # XLM-R / RoBERTa style special tokens
        self.bos_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

        if self.bos_token_id is None or self.eos_token_id is None or self.pad_token_id is None:
            raise ValueError(
                "Tokenizer must define cls_token_id, sep_token_id, and pad_token_id. "
                "XLM-R tokenizer is a good fit here."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _maybe_add_lang_token(self, text: str, lang: Optional[str]) -> str:
        if not self.add_lang_token or lang is None:
            return text
        return f"<{lang}> {text}"

    def _encode_source(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=self.max_source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def _encode_target(self, text: str) -> torch.Tensor:
        # For decoder training, we need special tokens so we can shift right.
        encoded = self.tokenizer(
            text,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].squeeze(0)
        return encoded

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        source_text = row[self.source_col]
        target_text = row[self.target_col]
        lang = row[self.lang_col] if self.lang_col is not None else None

        source_text = self._maybe_add_lang_token(source_text, lang)

        source = self._encode_source(source_text)
        target_ids = self._encode_target(target_text)

        # Teacher forcing:
        # decoder input  = <bos> ... tokens without the last token
        # labels         = tokens without the first token ... <eos>
        target_input_ids = target_ids[:-1]
        target_labels = target_ids[1:]

        # Since we removed one token on each side, pad back to max_target_len - 1
        # to keep batches rectangular.
        target_input_ids = torch.cat(
            [
                target_input_ids,
                torch.full((1,), self.pad_token_id, dtype=torch.long),
            ]
        )
        target_labels = torch.cat(
            [
                target_labels,
                torch.full((1,), self.pad_token_id, dtype=torch.long),
            ]
        )

        return {
            "source_ids": source["input_ids"],
            "source_mask": source["attention_mask"],
            "target_input_ids": target_input_ids,
            "target_labels": target_labels,
            "lang": lang,
        }


class GECDataCollator:
    """
    Collator that stacks already padded tensors from GECDataset.
    """

    def __call__(self, batch):
        source_ids = torch.stack([item["source_ids"] for item in batch])
        source_mask = torch.stack([item["source_mask"] for item in batch])
        target_input_ids = torch.stack([item["target_input_ids"] for item in batch])
        target_labels = torch.stack([item["target_labels"] for item in batch])

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_input_ids": target_input_ids,
            "target_labels": target_labels,
            "langs": [item.get("lang") for item in batch],
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    ds = GECDataset(
        csv_path="data/processed/train.csv",
        tokenizer=tokenizer,
        max_source_len=32,
        max_target_len=32,
    )
    sample = ds[0]
    print(sample["source_ids"].shape)
    print(sample["source_mask"].shape)
    print(sample["target_input_ids"].shape)
    print(sample["target_labels"].shape)
