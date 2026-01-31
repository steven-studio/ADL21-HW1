from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        texts = [s["text"] for s in samples]
        has_intent = "intent" in samples[0]

        if has_intent:
            intents = [s["intent"] for s in samples]
        ids = [s.get("id") for s in samples]

        # pad id
        pad_id = getattr(self.vocab, "pad_id", None)
        if pad_id is None:
            pad_id = 0

        def encode_one(text: str) -> List[int]:
            if hasattr(self.vocab, "encode"):
                out = self.vocab.encode(text)
                return out.tolist() if isinstance(out, torch.Tensor) else list(out)
            raise RuntimeError("Vocab must implement encode(text)")

        token_ids: List[List[int]] = []
        lengths: List[int] = []

        for t in texts:
            ids_ = encode_one(t)
            if self.max_len is not None and len(ids_) > self.max_len:
                ids_ = ids_[: self.max_len]
            token_ids.append(ids_)
            lengths.append(len(ids_))

        max_t = max(lengths) if lengths else 0
        input_ids = torch.full((len(samples), max_t), pad_id, dtype=torch.long)
        for i, ids_ in enumerate(token_ids):
            if ids_:
                input_ids[i, : len(ids_)] = torch.tensor(ids_, dtype=torch.long)

        lengths_t = torch.tensor(lengths, dtype=torch.long)
        if has_intent:
            label_ids = [self.label2idx(x) for x in intents]
            labels_t = torch.tensor(label_ids, dtype=torch.long)
        else:
            labels_t = None
            
        batch = {
            "input_ids": input_ids,
            "lengths": lengths_t,
            "ids": ids,
        }

        if labels_t is not None:
            batch["labels"] = labels_t

        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
