import pytest
import torch
from unittest.mock import MagicMock
from torch.utils.data import DataLoader

from src.rl.reward_model import RewardModel, PairwiseRewardLoss, RewardOutput
from src.rl.trainer import (
    TrainerConfig,
    CedarPairwiseDataset,
    RewardTrainer,
    collate_fn,
)
from src.rl.ppo import PPOTrainer, RLConfig, FeedbackDataset, Experience


class TestTrainerConfig:

    def test_defaults(self):
        config = TrainerConfig()
        assert config.vocab_size == 50000
        assert config.embed_dim == 256
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.dropout == 0.1
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.max_grad_norm == 1.0
        assert config.checkpoint_path is None

    def test_custom_config(self):
        config = TrainerConfig(vocab_size=1000, num_epochs=3, learning_rate=5e-4)
        assert config.vocab_size == 1000
        assert config.num_epochs == 3
        assert config.learning_rate == 5e-4

    def test_device_is_string(self):
        config = TrainerConfig()
        assert isinstance(config.device, str)
        assert config.device in ("cpu", "cuda")


class TestCedarPairwiseDataset:

    def setup_method(self):
        self.preferred = [[1, 2, 3], [4, 5, 6]]
        self.rejected  = [[7, 8, 9], [10, 11, 12]]
        self.dataset   = CedarPairwiseDataset(self.preferred, self.rejected)

    def test_length(self):
        assert len(self.dataset) == 2

    def test_getitem_keys(self):
        item = self.dataset[0]
        assert "preferred" in item
        assert "rejected" in item

    def test_getitem_types(self):
        item = self.dataset[0]
        assert isinstance(item["preferred"], torch.Tensor)
        assert isinstance(item["rejected"], torch.Tensor)

    def test_getitem_values(self):
        item = self.dataset[0]
        assert item["preferred"].tolist() == [1, 2, 3]
        assert item["rejected"].tolist() == [7, 8, 9]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            CedarPairwiseDataset([[1, 2]], [[3, 4], [5, 6]])


class TestCollateFn:

    def test_output_keys(self):
        batch = [
            {"preferred": torch.tensor([1, 2, 3]), "rejected": torch.tensor([4, 5])},
            {"preferred": torch.tensor([6, 7]),    "rejected": torch.tensor([8, 9, 10])},
        ]
        result = collate_fn(batch)
        assert "preferred"      in result
        assert "preferred_mask" in result
        assert "rejected"       in result
        assert "rejected_mask"  in result

    def test_padding(self):
        batch = [
            {"preferred": torch.tensor([1, 2, 3]), "rejected": torch.tensor([4, 5])},
            {"preferred": torch.tensor([6, 7]),    "rejected": torch.tensor([8, 9, 10])},
        ]
        result = collate_fn(batch)
        assert result["preferred"].shape == (2, 3)
        assert result["rejected"].shape  == (2, 3)

    def test_mask_dtype(self):
        batch = [
            {"preferred": torch.tensor([1, 2]), "rejected": torch.tensor([3, 4])},
        ]
        result = collate_fn(batch)
        assert result["preferred_mask"].dtype == torch.bool
        assert result["rejected_mask"].dtype  == torch.bool

    def test_mask_values(self):
        batch = [
            {"preferred": torch.tensor([1, 2, 3]), "rejected": torch.tensor([4, 5])},
            {"preferred": torch.tensor([6, 7]),    "rejected": torch.tensor([8, 9, 10])},
        ]
        result = collate_fn(batch)
        assert result["preferred_mask"][0].all()
        assert not result["preferred_mask"][1, 2]


class TestRewardTrainer:

    def setup_method(self):
        self.config  = TrainerConfig(vocab_size=100, embed_dim=16, hidden_dim=8, num_epochs=2, batch_size=2)
        self.trainer = RewardTrainer(self.config)

    def test_model_initialized(self):
        assert isinstance(self.trainer.model, RewardModel)

    def test_loss_fn_initialized(self):
        assert isinstance(self.trainer.loss_fn, PairwiseRewardLoss)

    def test_device(self):
        assert self.trainer.device == torch.device(self.config.device)

    def test_train_one_epoch(self):
        preferred = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        rejected  = [[13, 14], [15, 16], [17, 18], [19, 20]]
        dataset   = CedarPairwiseDataset(preferred, rejected)
        loader    = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        loss      = self.trainer._train_epoch(loader)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_eval_epoch(self):
        preferred = [[1, 2, 3], [4, 5, 6]]
        rejected  = [[7, 8], [9, 10]]
        dataset   = CedarPairwiseDataset(preferred, rejected)
        loader    = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        loss, acc = self.trainer._eval_epoch(loader)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_train_runs(self):
        preferred = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        rejected  = [[13, 14], [15, 16], [17, 18], [19, 20]]
        dataset   = CedarPairwiseDataset(preferred, rejected)
        loader    = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        self.trainer.train(loader)

    def test_train_with_val(self):
        preferred = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        rejected  = [[13, 14], [15, 16], [17, 18], [19, 20]]
        dataset   = CedarPairwiseDataset(preferred, rejected)
        loader    = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        self.trainer.train(loader, val_loader=loader)

    def test_score_response(self):
        score = self.trainer.score_response([1, 2, 3])
        assert isinstance(score, float)

    def test_rank_responses(self):
        candidates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ranking    = self.trainer.rank_responses(candidates)
        assert sorted(ranking) == [0, 1, 2]
        assert len(ranking) == 3

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "checkpoint.pt")
        self.trainer.save(path)
        self.trainer.load(path)

    def test_save_creates_file(self, tmp_path):
        import os
        path = str(tmp_path / "checkpoint.pt")
        self.trainer.save(path)
        assert os.path.exists(path)


class TestRLConfig:

    def test_defaults(self):
        config = RLConfig()
        assert config.learning_rate == 1e-5
        assert config.gamma == 0.99
        assert config.clip_epsilon == 0.2
        assert config.ppo_epochs == 4
        assert config.batch_size == 32


class TestPPOTrainer:

    def setup_method(self):
        self.trainer = PPOTrainer()

    def test_default_config(self):
        assert isinstance(self.trainer.config, RLConfig)

    def test_empty_experience_buffer(self):
        assert self.trainer.experience_buffer == []

    def test_compute_advantages_length(self):
        rewards = [1.0, 0.5, 0.0, 1.0]
        values  = [0.9, 0.4, 0.1, 0.8]
        advs    = self.trainer.compute_advantages(rewards, values)
        assert len(advs) == len(rewards)

    def test_compute_advantages_types(self):
        rewards = [1.0, 0.0]
        values  = [0.5, 0.0]
        advs    = self.trainer.compute_advantages(rewards, values)
        assert all(isinstance(a, float) for a in advs)

    def test_train_reward_model(self):
        reward_model = RewardModel(vocab_size=100, embed_dim=16, hidden_dim=8)

        tokenizer = MagicMock()
        tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids":      torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        train_data = [
            {"context": "Hello", "response": "Hi there!", "rating": 1},
            {"context": "Hello", "response": "Go away.",  "rating": 0},
            {"context": "What is AI?", "response": "AI is...", "rating": 1},
            {"context": "What is AI?", "response": "Idk.",     "rating": 0},
        ]

        metrics = self.trainer.train_reward_model(reward_model, train_data, tokenizer, epochs=1)
        assert "losses"      in metrics
        assert "avg_reward"  in metrics
        assert len(metrics["losses"]) == 1


class TestFeedbackDataset:

    def setup_method(self):
        self.tokenizer = MagicMock()
        self.tokenizer.side_effect = lambda text, **kwargs: {
            "input_ids":      torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        self.data = [
            {"context": "Hello", "response": "Hi!", "rating": 1},
            {"context": "Bye",   "response": "See you!", "rating": 0},
        ]
        self.dataset = FeedbackDataset(self.data, self.tokenizer)

    def test_length(self):
        assert len(self.dataset) == 2

    def test_getitem_keys(self):
        item = self.dataset[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "rating"         in item

    def test_rating_tensor(self):
        item = self.dataset[0]
        assert isinstance(item["rating"], torch.Tensor)
        assert item["rating"].item() == 1.0