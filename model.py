from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_class = num_class

        emb_dim = embeddings.size(1)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(self.encoder_output_size, num_class)
        self.drop = nn.Dropout(dropout)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # 支援兩種呼叫方式：
        # 1) forward(batch_dict)  (符合你的 scaffold)
        # 2) forward(input_ids)   (避免你 train loop 不小心傳錯)
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            lengths = batch.get("lengths", None)
        else:
            input_ids = batch
            lengths = None

        if lengths is None:
            # 以 padding=0 推長度（你已驗證 pad=0）
            lengths = (input_ids != 0).sum(dim=1)

        x = self.embed(input_ids)          # [B, T, E]
        x = self.drop(x)

        # pack padded sequence
        lengths_cpu = lengths.to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)
        # h_n: [num_layers * num_directions, B, hidden_size]

        if self.bidirectional:
            # 取最後一層的 forward/backward hidden concat
            # forward: -2, backward: -1
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2H]
        else:
            h_last = h_n[-1]  # [B, H]

        h_last = self.drop(h_last)
        logits = self.fc(h_last)           # [B, C]
        return {"logits": logits}


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
