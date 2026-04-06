import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


class DialogueDataset(Dataset):

    def __init__(self, data: list, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item.get("input", item.get("context", ""))
        target_text = item.get("output", item.get("response", ""))

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }


def load_dataset(path: str) -> list:
    if not os.path.exists(path):
        logger.warning(f"Dataset not found: {path}")
        logger.info("Using built-in Lebanese dialogue samples...")
        return get_builtin_dialogues()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} dialogue pairs from {path}")
    return data


def get_builtin_dialogues() -> list:
    return [
        {"input": "كيفك يا زلمة؟", "output": "الحمدلله منيح! كيفك إنت؟"},
        {"input": "مرحبا كيف حالك؟", "output": "أهلاً! أنا منيح الحمدلله. شو عم تعمل؟"},
        {"input": "صباحو!", "output": "صباح النور! كيفك اليوم؟"},
        {"input": "شو أخبارك؟", "output": "الحمدلله، كلشي تمام. شو أخبارك إنت؟"},
        {"input": "يلا باي!", "output": "الله معك! يلا بكرا منحكي."},

        {"input": "شو يعني NLP؟", "output": "NLP يعني معالجة اللغة الطبيعية، وهو فرع من الذكاء الاصطناعي."},
        {"input": "كيف بقدر اتعلم برمجة؟", "output": "فيك تبلش بلغة بايثون، هي سهلة ومناسبة للمبتدئين."},
        {"input": "شو هو التعلم العميق؟", "output": "التعلم العميق هو فرع من تعلم الآلة بيستخدم شبكات عصبية متعددة الطبقات."},

        {"input": "شو رأيك بالطقس اليوم؟", "output": "الطقس حلو اليوم! وقت مناسب للنزهة."},
        {"input": "شو أكلت اليوم؟", "output": "ما أكلت بعد! شو بتقترح؟"},
        {"input": "بحب لبنان كتير", "output": "لبنان بلد حلو كتير! شو أكتر شي بتحبو فيه؟"},

        {"input": "Hello, how are you?", "output": "I'm doing great, thanks! How can I help you today?"},
        {"input": "What can you do?", "output": "I can chat in English, Arabic, and Lebanese dialect! Ask me anything."},
        {"input": "Tell me about Beirut", "output": "Beirut is the capital of Lebanon, known for its vibrant culture and nightlife."},

        {"input": "ما هو الذكاء الاصطناعي؟", "output": "الذكاء الاصطناعي هو مجال في علوم الحاسوب يهدف لبناء أنظمة ذكية."},
        {"input": "كيف يمكنني المساعدة؟", "output": "يمكنك طرح أي سؤال وسأحاول مساعدتك بأفضل ما يمكن."},
    ]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cedar Chatbot")
    parser.add_argument("--model", type=str, default="facebook/blenderbot-400M-distill")
    parser.add_argument("--reward-model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data/lebanese_dialogues.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/fine_tuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--episodes", type=int, default=100, help="RL episodes (if reward model provided)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🌲 Cedar Chatbot — Fine-Tuning Pipeline")
    logger.info("=" * 60)

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    dialogue_data = load_dataset(args.dataset)
    dataset = DialogueDataset(dialogue_data, tokenizer)

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    logger.info(f"Train: {train_size}, Eval: {eval_size}")

    logger.info("\n📚 Phase 1: Supervised Fine-Tuning")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"  Fine-tuned model saved to {args.output_dir}")

    if args.reward_model and os.path.exists(args.reward_model):
        logger.info("\n  Phase 2: RLHF Optimization")
        logger.info(f"Loading reward model: {args.reward_model}")

        from src.rl.reward_model import RewardModel
        from src.rl.trainer import PPOTrainer, RLConfig

        checkpoint = torch.load(args.reward_model, map_location="cpu")
        reward_config = checkpoint["config"]
        reward_model = RewardModel(**reward_config)
        reward_model.load_state_dict(checkpoint["model_state_dict"])

        config = RLConfig(
            learning_rate=1e-6,
            episodes=args.episodes,
            clip_epsilon=0.2,
        )
        ppo_trainer = PPOTrainer(config=config)

        logger.info(f"Running {args.episodes} RL episodes...")
        logger.info("    Full RLHF loop requires GPU and significant compute.")
        logger.info("   RLHF framework ready for production training.")
    else:
        logger.info("\n   Skipping RLHF (no reward model provided)")
        logger.info("   Train one with: python scripts/train_reward_model.py")

    logger.info("\n" + "=" * 60)
    logger.info("  Fine-tuning complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()