import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def _set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, pad_tag_id: int = -100) -> Tuple[float, float]:
    """
    回傳 (avg_loss, token_acc)
    pad_tag_id: 用來忽略 padding token 的 label id（常見是 -100 或 tag_vocab["PAD"]）
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_id)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        # ===== 你可能需要依照 dataset 的 batch 格式做調整 =====
        # 常見格式：
        # batch = {
        #   "tokens": LongTensor [B, T],
        #   "tags": LongTensor [B, T],
        #   "lengths": LongTensor [B]
        # }
        tokens = batch["tokens"].to(device)
        tags = batch["tags"].to(device)

        # logits: [B, T, C]
        logits = model(tokens)

        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), tags.view(B * T))

        total_loss += loss.item()

        pred = logits.argmax(dim=-1)  # [B, T]
        mask = (tags != pad_tag_id)
        total_correct += (pred[mask] == tags[mask]).sum().item()
        total_count += mask.sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main(args):
    device = torch.device(args.device)

    train_path = args.data_dir / "train.json"
    dev_path = args.data_dir / "eval.json"

    # ===== 建 vocab（只用 train）=====
    print("Building vocab from train set...")
    data_train = json.loads(train_path.read_text())
    data_dev = json.loads(dev_path.read_text())

    all_tokens = []
    all_tags = []
    for ex in data_train:
        all_tokens.extend(ex["tokens"])
        all_tags.extend(ex["tags"])

    token_vocab = Vocab(sorted(set(all_tokens)))
    tag_vocab   = Vocab(sorted(set(all_tags)))

    # ===== 创建 embedding matrix =====
    # 选项1: 随机初始化
    embeddings = torch.randn(len(token_vocab), args.hidden_size)
    
    # 选项2: 如果有预训练 embeddings 文件，可以加载
    # embeddings = _load_pickle(args.cache_dir / "embeddings.pkl")

    # ===== 建 dataset =====
    train_set = SeqTaggingClsDataset(
        data_train,
        token_vocab,
        tag_vocab.token2idx,
        args.max_len,
    )

    dev_set = SeqTaggingClsDataset(
        data_dev,
        token_vocab,
        tag_vocab.token2idx,
        args.max_len,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dev_set.collate_fn,
    )

    # ===== model =====
    model = SeqTagger(
        embeddings,  # ← 改为传入 embedding matrix
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag_vocab),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ← 改为 -100

    best_acc = 0.0

    for epoch in range(args.num_epoch):
        # ===== train =====
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            tokens = batch["tokens"].to(device)
            tags = batch["tags"].to(device)

            logits = model(tokens)  # [B,T,C]
            B, T, C = logits.shape

            loss = criterion(
                logits.view(B*T, C),
                tags.view(B*T),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = tags != -100  # ← 改为 -100
                total_correct += (pred[mask] == tags[mask]).sum().item()
                total_tokens += mask.sum().item()

        train_acc = total_correct / total_tokens

        # ===== eval =====
        model.eval()
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="eval"):
                tokens = batch["tokens"].to(device)
                tags = batch["tags"].to(device)

                logits = model(tokens)
                pred = logits.argmax(dim=-1)

                mask = tags != -100  # ← 改为 -100
                total_correct += (pred[mask] == tags[mask]).sum().item()
                total_tokens += mask.sum().item()

        dev_acc = total_correct / total_tokens

        print(f"[Epoch {epoch}] train_acc={train_acc:.4f} dev_acc={dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "token_vocab": token_vocab,
                    "tag_vocab": tag_vocab,
                },
                args.ckpt_dir / "best.pt",
            )
            print("Saved best model.")            

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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