"""
============================================================
Agentic World — Fine-Tuning Job
============================================================
Fine-tunes Mistral Nemo 12B Instruct on behavioral session
descriptions using QLoRA (PEFT + BitsAndBytes).

Stack:
    - HuggingFace Transformers (model loading, tokenizer)
    - PEFT (LoRA adapters)
    - BitsAndBytes (4-bit quantization)
    - TRL SFTTrainer (supervised fine-tuning)
    - W&B (observation / tracking only)

Usage:
    # Local run with local data
    python finetune_job.py

    # Override hyperparams via env
    WANDB_PROJECT=agentic-world python finetune_job.py

    # With W&B data artifact
    python finetune_job.py  (set data_artifact in config)
============================================================
"""

import os
import json
import argparse
import torch
import wandb
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset as HFDataset

# ============================================================
# PARSE ARGS (allows overriding key params from CLI)
# ============================================================

parser = argparse.ArgumentParser(description="Fine-tune Mistral Nemo with QLoRA")
parser.add_argument("--train-file", type=str, default="train.jsonl")
parser.add_argument("--eval-file", type=str, default="eval.jsonl")
parser.add_argument("--model", type=str, default="mistralai/Mistral-Nemo-Instruct-2407")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=64)
parser.add_argument("--max-seq-length", type=int, default=2048)
parser.add_argument("--output-dir", type=str, default="outputs/mistral-nemo-behavioral-lora")
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
cli_args = parser.parse_args()

# ============================================================
# CONFIG
# ============================================================

# Help with CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CONFIG = {
    # Model
    "model_name": cli_args.model,
    "max_seq_length": cli_args.max_seq_length,
    "load_in_4bit": True,
    # LoRA
    "lora_rank": cli_args.lora_rank,
    "lora_alpha": cli_args.lora_alpha,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    # Training
    "num_epochs": cli_args.epochs,
    "batch_size": cli_args.batch_size,
    "gradient_accumulation": cli_args.gradient_accumulation,
    "learning_rate": cli_args.learning_rate,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "seed": 42,
    # Data
    "train_file": cli_args.train_file,
    "eval_file": cli_args.eval_file,
    # Output
    "output_dir": cli_args.output_dir,
}

# ============================================================
# W&B INIT (observation only)
# ============================================================

USE_WANDB = not cli_args.no_wandb

if USE_WANDB:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "agentic-world"),
        job_type="finetune",
        config=CONFIG,
    )
    print(f"W&B run: {run.name} ({run.id})")
else:
    run = None
    print("W&B disabled")

C = CONFIG  # use dict directly, no wandb.config dependency

print("=" * 60)
print("AGENTIC WORLD — FINE-TUNING")
print("=" * 60)
print(f"Model:  {C['model_name']}")
print(f"LoRA:   rank={C['lora_rank']}, alpha={C['lora_alpha']}")
print(f"Epochs: {C['num_epochs']}, LR: {C['learning_rate']}")
print(f"Data:   {C['train_file']}, {C['eval_file']}")
print("=" * 60)

# ============================================================
# LOAD MODEL (BitsAndBytes 4-bit quantization)
# ============================================================

print(f"\nLoading model: {C['model_name']}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
    print("  Using Flash Attention 2")
except ImportError:
    attn_impl = "sdpa"
    print("  Flash Attention 2 not found, using SDPA (torch native)")

model = AutoModelForCausalLM.from_pretrained(
    C["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(C["model_name"], trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# ============================================================
# PREPARE FOR KBIT TRAINING + APPLY LORA
# ============================================================

print(f"\nPreparing model for QLoRA training...")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=C["lora_rank"],
    lora_alpha=C["lora_alpha"],
    lora_dropout=C["lora_dropout"],
    target_modules=C["target_modules"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_pct = 100 * trainable_params / total_params
print(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
model.print_trainable_parameters()

if USE_WANDB:
    wandb.log({"trainable_params": trainable_params, "total_params": total_params, "trainable_pct": trainable_pct})

# ============================================================
# LOAD & PREPARE DATASET
# ============================================================

train_file = C["train_file"]
eval_file = C["eval_file"]

print(f"\nLoading dataset...")
print(f"  Train: {train_file}")
print(f"  Eval:  {eval_file}")

dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})

print(f"  Train examples: {len(dataset['train'])}")
print(f"  Eval examples:  {len(dataset['eval'])}")


def to_text(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


train_texts = [to_text(ex["messages"]) for ex in dataset["train"]]
eval_texts = [to_text(ex["messages"]) for ex in dataset["eval"]]

train_dataset = HFDataset.from_dict({"text": train_texts})
eval_dataset = HFDataset.from_dict({"text": eval_texts})

text_lengths = [len(t) for t in train_texts]
avg_len = sum(text_lengths) / len(text_lengths)
print(f"  Avg text length: {avg_len:.0f} chars")
print(f"  Min/Max: {min(text_lengths)}/{max(text_lengths)} chars")

if USE_WANDB:
    wandb.log({
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "avg_text_length": avg_len,
        "max_text_length": max(text_lengths),
        "min_text_length": min(text_lengths),
    })
    sample_table = wandb.Table(columns=["idx", "text_preview", "length"])
    for i, t in enumerate(train_texts[:5]):
        sample_table.add_data(i, t[:500], len(t))
    wandb.log({"data_samples": sample_table})

# ============================================================
# TRAIN
# ============================================================

output_dir = C["output_dir"]

print(f"\nStarting training...")
print(f"  Epochs: {C['num_epochs']}")
print(f"  Batch: {C['batch_size']} (effective: {C['batch_size'] * C['gradient_accumulation']})")
print(f"  LR: {C['learning_rate']}, Scheduler: {C['lr_scheduler']}")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=C["batch_size"],
        gradient_accumulation_steps=C["gradient_accumulation"],
        num_train_epochs=C["num_epochs"],
        learning_rate=C["learning_rate"],
        warmup_steps=C["warmup_steps"],
        weight_decay=C["weight_decay"],
        lr_scheduler_type=C["lr_scheduler"],
        seed=C["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="epoch",
        eval_accumulation_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="wandb" if USE_WANDB else "none",
        run_name=run.name if USE_WANDB else None,
        max_length=C["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
reserved_mem = torch.cuda.max_memory_reserved() / 1024**3
print(f"  GPU: {gpu_stats.name} ({gpu_stats.total_memory / 1024**3:.1f} GB)")
print(f"  Reserved: {reserved_mem:.1f} GB")

train_result = trainer.train()

peak_memory = torch.cuda.max_memory_reserved() / 1024**3
print(f"\nTraining complete!")
print(f"  Loss: {train_result.training_loss:.4f}")
print(f"  Time: {train_result.metrics['train_runtime']:.1f}s")
print(f"  Peak VRAM: {peak_memory:.1f} GB")

# ============================================================
# EVALUATE
# ============================================================

print(f"\nEvaluating...")
eval_results = trainer.evaluate()
print(f"  Eval loss: {eval_results['eval_loss']:.4f}")

if USE_WANDB:
    wandb.log({
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
        "peak_vram_gb": peak_memory,
    })

# ============================================================
# SAVE ADAPTER + LOG AS W&B ARTIFACT
# ============================================================

print(f"\nSaving adapter to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

config_path = os.path.join(output_dir, "training_config.json")
with open(config_path, "w") as f:
    json.dump({
        "base_model": C["model_name"],
        "lora_rank": C["lora_rank"],
        "lora_alpha": C["lora_alpha"],
        "lora_dropout": C["lora_dropout"],
        "target_modules": C["target_modules"],
        "epochs": C["num_epochs"],
        "learning_rate": C["learning_rate"],
        "lr_scheduler": C["lr_scheduler"],
        "max_seq_length": C["max_seq_length"],
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
        "peak_vram_gb": peak_memory,
        "wandb_run_id": run.id if USE_WANDB else None,
    }, f, indent=2)

if USE_WANDB:
    print(f"\nLogging adapter as W&B artifact...")
    adapter_artifact = wandb.Artifact(
        name="mistral-nemo-behavioral-lora",
        type="model",
        description=f"QLoRA adapter — train_loss={train_result.training_loss:.4f}, eval_loss={eval_results['eval_loss']:.4f}",
        metadata={
            "base_model": C["model_name"],
            "lora_rank": C["lora_rank"],
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_results["eval_loss"],
            "train_examples": len(train_dataset),
        },
    )
    adapter_artifact.add_dir(output_dir)
    run.log_artifact(adapter_artifact)
    print(f"  Artifact logged: mistral-nemo-behavioral-lora")

# ============================================================
# TEST INFERENCE
# ============================================================

print(f"\nRunning inference test...")

# Merge adapter for inference (or just use the PEFT model directly)
model.eval()

test_input = """Website: https://vinyl-vault.example.com/
Description: An online store selling vintage vinyl records. The homepage features a hero banner with staff picks, a genre filter sidebar (Jazz, Rock, Electronic, Classical, Hip-Hop), and a grid of album cards showing cover art, artist name, album title, price, and condition rating."""

messages = [
    {
        "role": "system",
        "content": "You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings.",
    },
    {"role": "user", "content": test_input},
]

input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*60}")
print("INFERENCE TEST — Vinyl Record Store")
print("=" * 60)
print(response[:2000])

if USE_WANDB:
    inference_table = wandb.Table(columns=["input", "output", "length"])
    inference_table.add_data(test_input[:300], response[:1000], len(response))
    wandb.log({"inference_test": inference_table})

# ============================================================
# DONE
# ============================================================

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Model:       {C['model_name']}")
print(f"  LoRA:        rank={C['lora_rank']}, alpha={C['lora_alpha']}")
print(f"  Train loss:  {train_result.training_loss:.4f}")
print(f"  Eval loss:   {eval_results['eval_loss']:.4f}")
print(f"  Time:        {train_result.metrics['train_runtime']:.1f}s")
print(f"  Peak VRAM:   {peak_memory:.1f} GB")
print(f"  Adapter:     {output_dir}/")
if USE_WANDB:
    print(f"  W&B run:     {run.url}")
print("=" * 60)

if USE_WANDB:
    wandb.finish()
