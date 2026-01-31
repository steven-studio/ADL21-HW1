import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    device = torch.device(args.device)
    
    # ===== 加载模型和词汇表 =====
    print(f"Loading model from {args.ckpt_dir / 'best.pt'}...")
    checkpoint = torch.load(args.ckpt_dir / "best.pt", map_location=device)
    
    token_vocab = checkpoint["token_vocab"]
    tag_vocab = checkpoint["tag_vocab"]
    
    # ===== 加载测试数据 =====
    test_path = args.data_dir / "test.json"
    print(f"Loading test data from {test_path}...")
    data_test = json.loads(test_path.read_text())
    
    # ===== 建立测试 dataset =====
    test_set = SeqTaggingClsDataset(
        data_test,
        token_vocab,
        tag_vocab.token2idx,
        args.max_len,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )
    
    # ===== 重建模型结构 =====
    embeddings = torch.randn(len(token_vocab), args.hidden_size)
    
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag_vocab),
    ).to(device)
    
    # ===== 加载模型权重 =====
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # ===== 预测 =====
    print("Predicting...")
    all_predictions = []
    all_ids = []
    
    # 创建 idx2tag 映射
    idx2tag = {idx: tag for tag, idx in tag_vocab.token2idx.items()}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            tokens = batch["tokens"].to(device)
            
            batch_size = tokens.size(0)
            start_idx = i * args.batch_size
            
            logits = model(tokens)  # [B, T, num_class]
            predictions = logits.argmax(dim=-1)  # [B, T]
            
            # 处理每个样本
            for j in range(batch_size):
                sample_idx = start_idx + j
                if sample_idx >= len(data_test):
                    break
                    
                # 获取原始数据
                sample = data_test[sample_idx]
                sample_id = sample.get("id", sample_idx)
                
                # 获取真实长度（去除 padding）
                pred_tags = predictions[j].cpu().numpy()
                num_tokens = len(sample["tokens"])
                
                # 将 tag id 转换为 tag 字符串
                tag_names = []
                for tag_id in pred_tags[:num_tokens]:
                    tag_name = idx2tag.get(int(tag_id), "O")
                    tag_names.append(tag_name)
                
                all_ids.append(sample_id)
                all_predictions.append(tag_names)
    
    # ===== 保存预测结果为 CSV =====
    print(f"Saving predictions to {args.pred_file}...")
    with open(args.pred_file, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(["id", "tags"])
        
        # 写入每个样本的预测
        for sample_id, tags in zip(all_ids, all_predictions):
            tags_str = " ".join(tags)
            writer.writerow([sample_id, tags_str])
    
    print(f"Prediction completed! Results saved to {args.pred_file}")


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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)