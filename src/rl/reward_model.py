import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class RewardOutput:
    score: float
    confidence: float


class RewardModel(nn.Module):

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        embeds = self.embedding(input_ids)
        lstm_out, _ = self.encoder(embeds)

        attn_weights = self.attention(lstm_out)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(), float("-inf")
            )
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (lstm_out * attn_weights).sum(dim=1)

        return self.reward_head(pooled)

    def score(self, input_ids: torch.Tensor) -> RewardOutput:
        self.eval()
        with torch.no_grad():
            reward = self.forward(input_ids.unsqueeze(0))
            score = reward.item()
        return RewardOutput(
            score=score,
            confidence=min(abs(score), 1.0),
        )


class PairwiseRewardLoss(nn.Module):

    def forward(
        self,
        preferred_score: torch.Tensor,
        rejected_score: torch.Tensor,
    ) -> torch.Tensor:
        return -torch.log(
            torch.sigmoid(preferred_score - rejected_score) + 1e-8
        ).mean()