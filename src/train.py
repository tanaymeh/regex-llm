import re
import signal

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from utils import soft_format_reward_func, correctness_reward_func


class TrainingConfig:
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    # MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    DATASET_PATH = "data/data.json"
    WANDB_PROJECT = "regex-r1"
    OUTPUT_DIR = "regex-r1-checkpoint"


def format_data(examples):
    prompts = []
    for p in examples["prompt"]:
        formatted = (
            "<|im_start|>system\n"
            "You are a coding expert specializing in Regular Expressions. "
            "Return only the valid regex pattern inside a code block.<|im_end|>\n"
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
    # max_length = 512  # Match your config
    # dataset = dataset.filter(
    #     lambda x: len(tokenizer(x["prompt"])["input_ids"]) <= max_length
    # )

    training_args = GRPOConfig(
        output_dir=TrainingConfig.OUTPUT_DIR,
        run_name=f"grpo-qwen-regex-{torch.cuda.device_count()}gpu",
        bf16=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,  # 24GB for vLLM, 24GB for Training
        num_generations=16,
        max_prompt_length=512,
        max_completion_length=128,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        generation_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=TrainingConfig.MODEL_NAME,
        reward_funcs=[soft_format_reward_func, correctness_reward_func],
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
