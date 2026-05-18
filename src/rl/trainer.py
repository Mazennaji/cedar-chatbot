import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from typing import Optional
import logging

from src.rl.reward_model import RewardModel, PairwiseRewardLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    vocab_size: int = 50000
    embed_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 10
    batch_size: int = 32
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None


class CedarPairwiseDataset(Dataset):

    def __init__(self, preferred: list[list[int]], rejected: list[list[int]]):
        assert len(preferred) == len(rejected)
        self.preferred = preferred
        self.rejected = rejected

    def __len__(self):
        return len(self.preferred)

    def __getitem__(self, idx):
        return {
            "preferred": torch.tensor(self.preferred[idx], dtype=torch.long),
            "rejected":  torch.tensor(self.rejected[idx], dtype=torch.long),
        }


def collate_fn(batch):
    preferred = [item["preferred"] for item in batch]
    rejected  = [item["rejected"]  for item in batch]
    preferred_padded = torch.nn.utils.rnn.pad_sequence(preferred, batch_first=True)
    rejected_padded  = torch.nn.utils.rnn.pad_sequence(rejected,  batch_first=True)
    return {
        "preferred":      preferred_padded,
        "preferred_mask": (preferred_padded != 0),
        "rejected":       rejected_padded,
        "rejected_mask":  (rejected_padded  != 0),
    }


class RewardTrainer:

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = RewardModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)

        self.loss_fn   = PairwiseRewardLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _forward_batch(self, batch):
        preferred      = batch["preferred"].to(self.device)
        preferred_mask = batch["preferred_mask"].to(self.device)
        rejected       = batch["rejected"].to(self.device)
        rejected_mask  = batch["rejected_mask"].to(self.device)
        preferred_score = self.model(preferred, preferred_mask)
        rejected_score  = self.model(rejected,  rejected_mask)
        return preferred_score, rejected_score

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            preferred_score, rejected_score = self._forward_batch(batch)
            loss = self.loss_fn(preferred_score, rejected_score)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for batch in loader:
                preferred_score, rejected_score = self._forward_batch(batch)
                loss = self.loss_fn(preferred_score, rejected_score)
                total_loss += loss.item()
                correct    += (preferred_score > rejected_score).sum().item()
                total      += preferred_score.size(0)

        return total_loss / len(loader), correct / total

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            log = f"Epoch {epoch}/{self.config.num_epochs} | train_loss={train_loss:.4f}"

            if val_loader:
                val_loss, accuracy = self._eval_epoch(val_loader)
                log += f" | val_loss={val_loss:.4f} | pairwise_accuracy={accuracy:.4f}"

            scheduler.step()
            logger.info(log)
            print(log)

            if self.config.checkpoint_path:
                self.save(self.config.checkpoint_path)

    def score_response(self, token_ids: list[int]) -> float:
        tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        output = self.model.score(tensor.squeeze(0))
        return output.score

    def rank_responses(self, candidates: list[list[int]]) -> list[int]:
        scores = [self.score_response(c) for c in candidates]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def save(self, path: str):
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.config,
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")