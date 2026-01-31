from html import parser
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path):
    return json.loads(path.read_text())


def load_vocab(cache_dir: Path) -> Vocab:
    with open(cache_dir / "vocab.pkl", "rb") as f:
        return pickle.load(f)


def load_intent2idx(cache_dir: Path) -> Dict[str, int]:
    return read_json(cache_dir / "intent2idx.json")


def build_datasets(
    data_dir: Path,
    vocab: Vocab,
    intent2idx: Dict[str, int],
    max_len: int,
) -> Dict[str, SeqClsDataset]:
    data_paths = {split: data_dir / f"{split}.json" for split in SPLITS}
    raw_data = {split: read_json(p) for split, p in data_paths.items()}

    return {
        split: SeqClsDataset(split_data, vocab, intent2idx, max_len)
        for split, split_data in raw_data.items()
    }
    
    
def build_dataloaders(
    datasets: Dict[str, SeqClsDataset],
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    NOTE:
    - 如果你的 SeqClsDataset 有自訂 collate_fn（常見於 padding），請在這裡加上：
        collate_fn=datasets[TRAIN].collate_fn
      或者從 dataset 模組 import 對應的 collate_fn。
    """
    # TODO: 如果需要 collate_fn，取消註解並修改
    # collate_fn = getattr(datasets[TRAIN], "collate_fn", None)

    loaders: Dict[str, DataLoader] = {}

    loaders[TRAIN] = DataLoader(
        datasets[TRAIN],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # collate_fn=collate_fn,
    )
    loaders[DEV] = DataLoader(
        datasets[DEV],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # collate_fn=collate_fn,
    )
    return loaders


def save_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)


# -----------------------------
# Train / Eval loops (skeleton)
# -----------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    回傳 accuracy（0~1）。
    這裡不猜你的 batch 格式與 model output，所以留 TODO。
    """
    model.eval()

    correct = 0
    total = 0

    for batch in dataloader:
        # TODO: 將 batch 搬到 device
        # 例：
        input_ids = batch["input_ids"].to(device)
        lengths   = batch["lengths"].to(device)
        labels    = batch["labels"].to(device)

        # TODO: forward 得 logits（shape: [B, C]）
        logits = model(input_ids, lengths)

        # TODO: 取 prediction，更新 correct/total
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        pass

    return (correct / total) if total > 0 else 0.0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: torch.nn.Module,   # ← 加這行
) -> float:
    """
    回傳平均 loss。
    這裡不猜你的 batch 格式與 model output，所以留 TODO。
    """
    model.train()

    total_loss = 0.0
    steps = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        # TODO: 將 batch 搬到 device
        input_ids = batch["input_ids"].to(device)
        lengths   = batch["lengths"].to(device)
        labels    = batch["labels"].to(device)

        # TODO: forward 得 logits
        # TODO: 算 loss (e.g., CrossEntropyLoss)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        pass

    return (total_loss / steps) if steps > 0 else 0.0


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    loaders = build_dataloaders(datasets, args.batch_size, num_workers=args.num_workers)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    from model import SeqClassifier
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
    )
    
    device = args.device
    model.to(device)

    # TODO: init optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_acc = -1.0
    best_ckpt_path = args.ckpt_dir / "best.pt"

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # train
        train_loss = train_one_epoch(model, loaders[TRAIN], optimizer, device, criterion)

        # eval
        dev_acc = evaluate(model, loaders[DEV], device)

        # save best
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_checkpoint(model, best_ckpt_path)

        epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            dev_acc=f"{dev_acc:.4f}",
            best=f"{best_dev_acc:.4f}",
        )

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # workers
    parser.add_argument("--num_workers", type=int, default=0)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
