import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.rl.reward_model import RewardModel
from src.rl.trainer import PPOTrainer, RLConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_feedback_data(path: str) -> list:
    if not os.path.exists(path):
        logger.warning(f"Feedback file not found: {path}")
        logger.info("Generating synthetic training data...")
        return generate_synthetic_data()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} feedback records from {path}")
    return data


def generate_synthetic_data() -> list:
    data = []

    good_responses = [
        {"context": "Hello!", "response": "Hi there! How can I help you today?", "rating": 1},
        {"context": "What is NLP?", "response": "NLP stands for Natural Language Processing, a field of AI.", "rating": 1},
        {"context": "كيف حالك؟", "response": "أنا بخير، شكراً! كيف يمكنني مساعدتك؟", "rating": 1},
        {"context": "keefak?", "response": "مرحبا! أنا منيح، شكراً لسؤالك", "rating": 1},
        {"context": "Tell me about Lebanon", "response": "Lebanon is a beautiful country in the Middle East known for its rich culture.", "rating": 1},
    ]

    bad_responses = [
        {"context": "Hello!", "response": "I don't understand.", "rating": -1},
        {"context": "What is NLP?", "response": "NLP NLP NLP", "rating": -1},
        {"context": "كيف حالك؟", "response": "error error error", "rating": -1},
        {"context": "Help me", "response": "", "rating": -1},
        {"context": "Good morning", "response": "Goodbye forever.", "rating": -1},
    ]

    data.extend(good_responses * 10)  # Oversample good
    data.extend(bad_responses * 10)

    logger.info(f"Generated {len(data)} synthetic feedback records")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train RLHF Reward Model")
    parser.add_argument("--data", type=str, default="data/feedback.json", help="Feedback data path")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="checkpoints/reward_model.pt", help="Output path")
    parser.add_argument("--model", type=str, default="facebook/blenderbot-400M-distill", help="Tokenizer model")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🌲 Cedar Chatbot — Reward Model Training")
    logger.info("=" * 60)

    feedback_data = load_feedback_data(args.data)

    if len(feedback_data) < 10:
        logger.error("Not enough feedback data (minimum 10 records)")
        sys.exit(1)

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    vocab_size = tokenizer.vocab_size
    reward_model = RewardModel(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in reward_model.parameters())
    logger.info(f"Reward model parameters: {total_params:,}")

    config = RLConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    trainer = PPOTrainer(config=config)

    logger.info(f"Training for {args.epochs} epochs...")
    metrics = trainer.train_reward_model(
        reward_model=reward_model,
        train_data=feedback_data,
        tokenizer=tokenizer,
        epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "embed_dim": 256,
            "hidden_dim": 128,
            "num_layers": 2,
        },
        "metrics": metrics,
    }, args.output)

    logger.info(f"✅ Reward model saved to {args.output}")
    logger.info(f"Final loss: {metrics['losses'][-1]:.4f}")
    logger.info(f"Final avg reward: {metrics['avg_reward'][-1]:.4f}")


if __name__ == "__main__":
    main()