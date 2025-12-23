import re
import signal

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from utils import (
    soft_format_reward_func,
    correctness_reward_func,
    regex_similarity_reward_func,
    name_generator,
    LoggingRewardCallback,
    reasoning_length_penalty_func
)


class TrainingConfig:
    MODEL_NAME = "Qwen/Qwen3-4B"
    # MODEL_NAME = "Qwen/Qwen3-1.7B"
    DATASET_PATH = "data/data.json"
    WANDB_PROJECT = "regex-r1"
    OUTPUT_DIR = "regex-r1-checkpoint"


def format_data(examples):
    prompts = []
    for p in examples["prompt"]:
        formatted = (
            "<|im_start|>system\n"
            "You are a coding expert specializing in Regular Expressions. "
            "Please reason step by step, and put your final answer within a ```regex ... ``` block. Only give ONE answer.<|im_end|>\n"
            "Keep your reasoning concise, short and effective. No need to state the obvious."
            "<|im_start|>user\n"
            f"{p}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(formatted)
    return {"prompt": prompts}


def main():
    print(f"Loading dataset from {TrainingConfig.DATASET_PATH}...")
    dataset = load_dataset(
        "json", data_files=TrainingConfig.DATASET_PATH, split="train"
    )

    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"Loading tokenizer for {TrainingConfig.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format the prompts
    train_dataset = train_dataset.map(format_data, batched=True)
    eval_dataset = eval_dataset.map(format_data, batched=True)

    reward_funcs = [
        soft_format_reward_func,
        regex_similarity_reward_func,
        correctness_reward_func,
        reasoning_length_penalty_func
    ]

    logging_callback = LoggingRewardCallback(reward_funcs)
    reward_funcs = reward_funcs + [logging_callback]

    training_args = GRPOConfig(
        output_dir=TrainingConfig.OUTPUT_DIR,
        run_name=f"{TrainingConfig.MODEL_NAME}_grpo_{name_generator()}_run",
        bf16=True,
        # optim="adamw_torch",  # Standard AdamW is fine with 140GB (faster than 8-bit)
        num_generations=32,
        per_device_train_batch_size=2,
        generation_batch_size=32,
        per_device_eval_batch_size=32,
        # Effective Batch Size = 32 (device) * 2 (accum) = 64
        gradient_accumulation_steps=8,
        max_prompt_length=512,
        max_completion_length=1024,
        learning_rate=5e-6,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=1,
        report_to="wandb",
        beta=0.001,
        temperature=0.8,
        project=TrainingConfig.WANDB_PROJECT,
    )

    trainer = GRPOTrainer(
        model=TrainingConfig.MODEL_NAME,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving model to {TrainingConfig.OUTPUT_DIR}...")
    trainer.save_model(TrainingConfig.OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
    tokenizer.save_pretrained(TrainingConfig.OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
