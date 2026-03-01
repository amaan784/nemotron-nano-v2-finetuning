"""
============================================================
Agentic World — Fine-Tuning Job (W&B Launch)
============================================================
This script is designed to run on a remote GPU via W&B Launch.
It pulls training data from a W&B artifact, fine-tunes Mistral
Nemo 12B with QLoRA, and logs the adapter back as an artifact.

Can also be run standalone:
    python finetune_job.py

All hyperparameters come from wandb.config (set by Launch or defaults).
============================================================
"""

import os
import json
import torch
import wandb
from pathlib import Path
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset as HFDataset

# ============================================================
# W&B INIT — config comes from Launch override or defaults
# ============================================================

DEFAULTS = {
    # Model
    "model_name": "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "max_seq_length": 4096,
    "load_in_4bit": True,
    # LoRA
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Training
    "num_epochs": 5,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "seed": 42,
    # Data artifact
    "data_artifact": None,  # e.g. "agentic-world/training-data:latest"
    "train_file": "train.jsonl",
    "eval_file": "eval.jsonl",
    # Output
    "output_dir": "outputs/mistral-nemo-behavioral-lora",
    "push_to_hf": None,  # HF repo name, e.g. "org/model-name"
}

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "agentic-world"),
    job_type="finetune",
    config=DEFAULTS,
)
C = wandb.config

print("=" * 60)
print("AGENTIC WORLD — FINE-TUNING JOB")
print("=" * 60)
print(f"Run:    {run.name} ({run.id})")
print(f"Model:  {C.model_name}")
print(f"LoRA:   rank={C.lora_rank}, alpha={C.lora_alpha}")
print(f"Epochs: {C.num_epochs}, LR: {C.learning_rate}")
print(f"Data:   {C.data_artifact or 'local files'}")
print("=" * 60)

# ============================================================
# PULL DATA FROM W&B ARTIFACT (or use local files)
# ============================================================

if C.data_artifact:
    print(f"\nPulling training data from artifact: {C.data_artifact}")
    artifact = run.use_artifact(C.data_artifact)
    data_dir = Path(artifact.download())
    train_file = str(data_dir / C.train_file)
    eval_file = str(data_dir / C.eval_file)
    print(f"  Train: {train_file}")
    print(f"  Eval:  {eval_file}")
else:
    train_file = C.train_file
    eval_file = C.eval_file
    print(f"\nUsing local data files: {train_file}, {eval_file}")

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model: {C.model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=C.model_name,
    max_seq_length=C.max_seq_length,
    dtype=None,
    load_in_4bit=C.load_in_4bit,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")

# ============================================================
# APPLY LORA
# ============================================================

print(f"\nApplying LoRA (rank={C.lora_rank}, alpha={C.lora_alpha})")
model = FastLanguageModel.get_peft_model(
    model,
    r=C.lora_rank,
    target_modules=list(C.target_modules),
    lora_alpha=C.lora_alpha,
    lora_dropout=C.lora_dropout,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=C.seed,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_pct = 100 * trainable_params / total_params
print(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")

wandb.log({"trainable_params": trainable_params, "total_params": total_params, "trainable_pct": trainable_pct})

# ============================================================
# LOAD & PREPARE DATASET
# ============================================================

print(f"\nLoading dataset...")
dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})

print(f"  Train: {len(dataset['train'])} examples")
print(f"  Eval:  {len(dataset['eval'])} examples")


def to_text(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


train_texts = [to_text(ex["messages"]) for ex in dataset["train"]]
eval_texts = [to_text(ex["messages"]) for ex in dataset["eval"]]

train_dataset = HFDataset.from_dict({"text": train_texts})
eval_dataset = HFDataset.from_dict({"text": eval_texts})

text_lengths = [len(t) for t in train_texts]
avg_len = sum(text_lengths) / len(text_lengths)

wandb.log({
    "train_examples": len(train_dataset),
    "eval_examples": len(eval_dataset),
    "avg_text_length": avg_len,
    "max_text_length": max(text_lengths),
    "min_text_length": min(text_lengths),
})

# Log a sample as a W&B Table
sample_table = wandb.Table(columns=["idx", "text_preview", "length"])
for i, t in enumerate(train_texts[:5]):
    sample_table.add_data(i, t[:500], len(t))
wandb.log({"data_samples": sample_table})

# ============================================================
# TRAIN
# ============================================================

print(f"\nStarting training...")
print(f"  Epochs: {C.num_epochs}")
print(f"  Batch: {C.batch_size} (effective: {C.batch_size * C.gradient_accumulation})")
print(f"  LR: {C.learning_rate}, Scheduler: {C.lr_scheduler}")

output_dir = C.output_dir

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=C.batch_size,
        gradient_accumulation_steps=C.gradient_accumulation,
        num_train_epochs=C.num_epochs,
        learning_rate=C.learning_rate,
        warmup_steps=C.warmup_steps,
        weight_decay=C.weight_decay,
        lr_scheduler_type=C.lr_scheduler,
        seed=C.seed,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run.name,
        max_length=C.max_seq_length,
        dataset_text_field="text",
        packing=False,
        dataset_num_proc=1,
        dataloader_num_workers=0,
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

# Save training config alongside adapter
config_path = os.path.join(output_dir, "training_config.json")
with open(config_path, "w") as f:
    json.dump({
        "base_model": C.model_name,
        "lora_rank": C.lora_rank,
        "lora_alpha": C.lora_alpha,
        "lora_dropout": C.lora_dropout,
        "target_modules": list(C.target_modules),
        "epochs": C.num_epochs,
        "learning_rate": C.learning_rate,
        "lr_scheduler": C.lr_scheduler,
        "max_length": C.max_seq_length,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
        "peak_vram_gb": peak_memory,
        "wandb_run_id": run.id,
        "wandb_run_name": run.name,
    }, f, indent=2)

# Log adapter as W&B artifact
print(f"\nLogging adapter as W&B artifact...")
adapter_artifact = wandb.Artifact(
    name="mistral-nemo-behavioral-lora",
    type="model",
    description=f"QLoRA adapter — train_loss={train_result.training_loss:.4f}, eval_loss={eval_results['eval_loss']:.4f}",
    metadata={
        "base_model": C.model_name,
        "lora_rank": C.lora_rank,
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "train_examples": len(train_dataset),
    },
)
adapter_artifact.add_dir(output_dir)
run.log_artifact(adapter_artifact)
print(f"  Artifact logged: mistral-nemo-behavioral-lora")

# ============================================================
# PUSH TO HF (optional)
# ============================================================

if C.push_to_hf:
    print(f"\nPushing to HuggingFace: {C.push_to_hf}")
    model.push_to_hub(C.push_to_hf, tokenizer=tokenizer)
    print(f"  Pushed!")

# ============================================================
# TEST INFERENCE
# ============================================================

print(f"\nRunning inference test...")
FastLanguageModel.for_inference(model)

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
).to("cuda")

output = model.generate(
    input_ids=input_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True,
)
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*60}")
print("INFERENCE TEST — Vinyl Record Store")
print("=" * 60)
print(response[:2000])

# Log inference test to W&B
inference_table = wandb.Table(columns=["input", "output", "length"])
inference_table.add_data(test_input[:300], response[:1000], len(response))
wandb.log({"inference_test": inference_table})

# ============================================================
# DONE
# ============================================================

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Model:       {C.model_name}")
print(f"  LoRA:        rank={C.lora_rank}, alpha={C.lora_alpha}")
print(f"  Train loss:  {train_result.training_loss:.4f}")
print(f"  Eval loss:   {eval_results['eval_loss']:.4f}")
print(f"  Time:        {train_result.metrics['train_runtime']:.1f}s")
print(f"  Peak VRAM:   {peak_memory:.1f} GB")
print(f"  Adapter:     {output_dir}/")
print(f"  W&B run:     {run.url}")
print("=" * 60)

wandb.finish()
