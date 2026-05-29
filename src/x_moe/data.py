import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tqdm import tqdm


def train_tokenizer(texts, vocab_size=4096, save_path="tokenizer.json", force=False):
    if save_path and not force and os.path.exists(save_path):
        return Tokenizer.from_file(save_path)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_seq_len=256, cache=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts
        self.eos_id = tokenizer.token_to_id("<eos>")
        self._cache = cache
        if cache:
            self._encoded = [self._encode(text) for text in tqdm(texts)]

    def _encode(self, text):
        encoded = self.tokenizer.encode(text).ids
        if len(encoded) > self.max_seq_len - 1:
            encoded = encoded[: self.max_seq_len - 1]
        tokens = encoded + [self.eos_id]
        tokens = tokens[: self.max_seq_len]
        return tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self._cache:
            tokens = self._encoded[idx]
        else:
            tokens = self._encode(self.texts[idx])
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, pad_id=0, pad_to_max=False, max_seq_len=None):
    if pad_to_max and max_seq_len is not None:
        padded_batch = []
        for b in batch:
            pad_len = max_seq_len - b.shape[0]
            if pad_len > 0:
                padded_batch.append(F.pad(b, (0, pad_len), value=pad_id))
            else:
                padded_batch.append(b[:max_seq_len] if b.shape[0] > max_seq_len else b)
        return torch.stack(padded_batch)
    max_len = max(b.shape[0] for b in batch)
    padded = []
    for b in batch:
        pad_len = max_len - b.shape[0]
        if pad_len > 0:
            padded.append(F.pad(b, (0, pad_len), value=pad_id))
        else:
            padded.append(b)
    return torch.stack(padded)


def get_collate_fn(pad_id=0, pad_to_max=False, max_seq_len=None):
    def _collate(batch):
        return collate_fn(
            batch, pad_id=pad_id, pad_to_max=pad_to_max, max_seq_len=max_seq_len
        )

    return _collate


def get_pad_eos_ids(tokenizer):
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    return pad_id, eos_id


def create_dataloaders(
    train_dataset,
    val_dataset=None,
    batch_size=32,
    num_workers=24,
    pad_id=0,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=get_collate_fn(pad_id=pad_id),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0 if train_loader else False,
        )
    return train_loader, val_loader
