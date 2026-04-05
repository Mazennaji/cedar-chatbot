import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    learning_rate: float = 1e-5
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 32
    episodes: int = 1000
    reward_baseline: float = 0.0


@dataclass
class Experience:
    state: torch.Tensor
    action: torch.Tensor  
    reward: float 
    log_prob: float
    value: float 


class FeedbackDataset(Dataset):

    def __init__(self, feedback_data: list, tokenizer, max_length: int = 128):
        self.data = feedback_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item.get("context", "")
        response = item.get("response", "")
        rating = item.get("rating", 0)

        combined = f"{context} {response}"
        encoded = self.tokenizer(
            combined,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "rating": torch.tensor(rating, dtype=torch.float),
        }


class PPOTrainer:

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.experience_buffer: list[Experience] = []

    def train_reward_model(
        self,
        reward_model: nn.Module,
        train_data: list,
        tokenizer,
        epochs: int = 5,
    ) -> dict:
        from src.rl.reward_model import PairwiseRewardLoss

        dataset = FeedbackDataset(train_data, tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        optimizer = optim.AdamW(
            reward_model.parameters(),
            lr=self.config.learning_rate,
        )
        criterion = PairwiseRewardLoss()

        metrics = {"losses": [], "avg_reward": []}
        reward_model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_rewards = []
            num_batches = 0

            for batch in loader:
                input_ids = batch["input_ids"]
                ratings = batch["rating"]

                scores = reward_model(input_ids).squeeze(-1)

                if len(scores) >= 2:
                    sorted_idx = ratings.argsort(descending=True)
                    preferred = scores[sorted_idx[::2]]
                    rejected = scores[sorted_idx[1::2]]
                    min_len = min(len(preferred), len(rejected))

                    if min_len > 0:
                        loss = criterion(
                            preferred[:min_len],
                            rejected[:min_len],
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            reward_model.parameters(),
                            self.config.max_grad_norm,
                        )
                        optimizer.step()

                        epoch_loss += loss.item()
                        epoch_rewards.extend(scores.detach().tolist())
                        num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_reward = (
                sum(epoch_rewards) / len(epoch_rewards)
                if epoch_rewards else 0
            )

            metrics["losses"].append(avg_loss)
            metrics["avg_reward"].append(avg_reward)

            logger.info(
                f"Epoch {epoch+1}/{epochs} — "
                f"Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}"
            )

        return metrics

    def compute_advantages(
        self,
        rewards: list[float],
        values: list[float],
    ) -> list[float]:
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = (
                rewards[t]
                + self.config.gamma * next_value
                - values[t]
            )
            gae = delta + self.config.gamma * 0.95 * gae
            advantages.insert(0, gae)

        return advantages

    def ppo_update(
        self,
        policy_model: nn.Module,
        optimizer: optim.Optimizer,
        old_log_probs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:

        total_loss = 0.0

        for _ in range(self.config.ppo_epochs):
            new_log_probs = policy_model(states, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            )
            policy_loss = -torch.min(
                ratio * advantages,
                clipped * advantages,
            ).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                self.config.max_grad_norm,
            )
            optimizer.step()

            total_loss += policy_loss.item()

        return total_loss / self.config.ppo_epochs