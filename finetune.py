"""
============================================================
Agentic World — Fine-Tuning Pipeline
============================================================
Fine-tunes NVIDIA Nemotron Nano 9B v2 on behavioral session
descriptions using Unsloth + QLoRA.

Model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
Method: QLoRA (4-bit quantization + LoRA adapters)
Framework: Unsloth + HuggingFace TRL
Tracking: Weights & Biases

Requirements (install on Brev instance):
    pip install unsloth
    pip install wandb datasets trl
    # Unsloth handles torch/transformers/bitsandbytes

Usage:
    python finetune.py
    
GPU: A10G (24GB) or A100 (40GB)
Expected VRAM: ~10-14GB with QLoRA
Expected time: 10-20 minutes for 37 examples
============================================================
"""

import os
import json
import torch
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

# Model
MODEL_NAME = "unsloth/NVIDIA-Nemotron-Nano-9B-v2-bnb-4bit"  # Pre-quantized 4-bit
MAX_SEQ_LENGTH = 4096  # Enough for ~1200 token outputs
DTYPE = None  # Auto-detect (float16 on T4/A10G, bfloat16 on A100)
LOAD_IN_4BIT = True

# LoRA
LORA_RANK = 32          # Higher rank = more capacity for behavioral nuance
LORA_ALPHA = 64         # Usually 2x rank
LORA_DROPOUT = 0.05
TARGET_MODULES = [       # All linear layers for maximum adaptation
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 1           # Small dataset, keep batch size 1
GRADIENT_ACCUMULATION = 4  # Effective batch size = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42

# Data
TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"

# Output
OUTPUT_DIR = "outputs/nemotron-behavioral-lora"
HF_REPO = None  # Set to "your-org/model-name" to push to HF

# W&B
WANDB_PROJECT = "agentic-world"
WANDB_RUN_NAME = f"nemotron-nano-behavioral-{datetime.now().strftime('%Y%m%d-%H%M')}"

# ============================================================
# SETUP W&B
# ============================================================

print("=" * 60)
print("AGENTIC WORLD — BEHAVIORAL MODEL FINE-TUNING")
print("=" * 60)

try:
    import wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "method": "QLoRA",
        },
        tags=["hackathon", "mistral-worldwide", "nvidia", "behavioral-finetuning"],
    )
    USE_WANDB = True
    print(f"W&B initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
except Exception as e:
    print(f"WARNING: W&B not available ({e}), continuing without tracking")
    USE_WANDB = False

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}")
print(f"   4-bit quantization: {LOAD_IN_4BIT}")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"Model loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# APPLY LORA
# ============================================================

print(f"\nApplying LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized checkpointing
    random_state=SEED,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"LoRA applied")
print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

if USE_WANDB:
    wandb.log({
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
    })

# ============================================================
# LOAD DATASET
# ============================================================

print(f"\nLoading dataset")
print(f"   Train: {TRAIN_FILE}")
print(f"   Eval:  {EVAL_FILE}")

from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "eval": EVAL_FILE,
})

print(f"   Train examples: {len(dataset['train'])}")
print(f"   Eval examples:  {len(dataset['eval'])}")

# Format function: apply chat template
def formatting_func(examples):
    """Convert messages to tokenized chat format."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

# Apply formatting
train_dataset = dataset["train"].map(formatting_func, batched=True, remove_columns=["messages"])
eval_dataset = dataset["eval"].map(formatting_func, batched=True, remove_columns=["messages"])

# Log sample
print(f"\nSample training example (first 300 chars):")
print(f"   {train_dataset[0]['text'][:300]}...")

if USE_WANDB:
    wandb.log({
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "avg_text_length": sum(len(t) for t in train_dataset["text"]) / len(train_dataset),
    })

# ============================================================
# TRAINING
# ============================================================

print(f"\nStarting training")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Scheduler: {LR_SCHEDULER}")

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if USE_WANDB else "none",
        run_name=WANDB_RUN_NAME if USE_WANDB else None,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # Don't pack, our examples are long
    ),
)

# Print GPU memory before training
gpu_stats = torch.cuda.get_device_properties(0)
reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
print(f"\n   GPU: {gpu_stats.name}")
print(f"   Total VRAM: {gpu_stats.total_memory / 1024 / 1024 / 1024:.1f} GB")
print(f"   Reserved: {reserved_memory:.1f} GB")

# Train
train_result = trainer.train()

print(f"\nTraining complete!")
print(f"   Training loss: {train_result.training_loss:.4f}")
print(f"   Training time: {train_result.metrics['train_runtime']:.1f}s")

# ============================================================
# EVALUATION
# ============================================================

print(f"\nRunning evaluation on held-out set...")

eval_results = trainer.evaluate()
print(f"   Eval loss: {eval_results['eval_loss']:.4f}")

if USE_WANDB:
    wandb.log({
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
    })

# ============================================================
# SAVE ADAPTER
# ============================================================

print(f"\nSaving LoRA adapter to {OUTPUT_DIR}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training config alongside
config_path = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_path, "w") as f:
    json.dump({
        "base_model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
    }, f, indent=2)

print(f"Adapter saved ({os.path.getsize(os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')) / 1024 / 1024:.1f} MB)")

# ============================================================
# PUSH TO HUGGING FACE (optional)
# ============================================================

if HF_REPO:
    print(f"\nPushing to Hugging Face: {HF_REPO}")
    model.push_to_hub(HF_REPO, tokenizer=tokenizer)
    print(f"Pushed to {HF_REPO}")

# ============================================================
# LOG ADAPTER AS W&B ARTIFACT
# ============================================================

if USE_WANDB:
    print(f"\nLogging adapter as W&B artifact...")
    artifact = wandb.Artifact(
        name="nemotron-behavioral-lora",
        type="model",
        description="QLoRA adapter for behavioral simulation on Nemotron Nano 9B v2",
        metadata={
            "base_model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "final_eval_loss": eval_results["eval_loss"],
        },
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
    print(f"Artifact logged to W&B")

# ============================================================
# TEST INFERENCE
# ============================================================

print(f"\nTesting inference with fine-tuned model...")

FastLanguageModel.for_inference(model)

# Test with a NEW website description (not FunCity)
test_input = """Website: https://vinyl-vault.example.com/
Description: An online store selling vintage vinyl records. The homepage features a hero banner with staff picks, a genre filter sidebar (Jazz, Rock, Electronic, Classical, Hip-Hop), and a grid of album cards showing cover art, artist name, album title, price, and condition rating (Mint/VG+/VG/Good). Users can click albums for detail pages with tracklists, seller reviews, and an Add to Cart button. There's a search bar in the top navigation, a wishlist feature, and a cart icon showing item count. New arrivals section at the bottom of the homepage."""

messages = [
    {
        "role": "system",
        "content": "You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."
    },
    {
        "role": "user",
        "content": test_input,
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

output = model.generate(
    input_ids=input_ids,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"INFERENCE TEST — New Website (Vinyl Record Store)")
print(f"{'='*60}")
print(response[:2000])
print(f"\n{'='*60}")
print(f"Response length: {len(response)} chars, ~{len(response)//4} tokens")

if USE_WANDB:
    wandb.log({
        "test_response": response,
        "test_response_length": len(response),
    })

# ============================================================
# ALSO TEST ON FUNCITY (sanity check)
# ============================================================

print(f"\nSanity check -- FunCity (training domain)...")

funcity_input = """Website: https://fun-city-xi.vercel.app/
Description: FunCity is a Reddit-style NYC discovery board where users browse posts organized by NYC boroughs (The Bronx, Brooklyn, Manhattan, Queens, Staten Island) and topics (Art & Culture, Food & Eats, Hidden Gems, Nature & Parks, Nightlife). The homepage shows a feed of user posts sorted by Hot, New, or Top tabs. Each post card displays a borough tag, username, timestamp, title, body preview, upvote/downvote arrows with score, and comment count. The right sidebar contains borough filter buttons, topic filter buttons, and a Trending section showing top 5 posts. Users can click posts to see the full post detail page with a comments thread (each comment has upvote/downvote). There is a Sign Up button (top right) with a modal collecting username, password, age group, country, and NYC familiarity. Logged-in users see a "+ New Post" button and can comment and vote."""

messages_fc = [
    {
        "role": "system",
        "content": "You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."
    },
    {
        "role": "user",
        "content": funcity_input,
    },
]

input_ids_fc = tokenizer.apply_chat_template(
    messages_fc,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

output_fc = model.generate(
    input_ids=input_ids_fc,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response_fc = tokenizer.decode(output_fc[0][input_ids_fc.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"SANITY CHECK — FunCity")
print(f"{'='*60}")
print(response_fc[:2000])

if USE_WANDB:
    wandb.log({
        "funcity_response": response_fc,
        "funcity_response_length": len(response_fc),
    })
    wandb.finish()

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — SUMMARY")
print(f"{'='*60}")
print(f"Model:            {MODEL_NAME}")
print(f"Method:           QLoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
print(f"Train examples:   {len(train_dataset)}")
print(f"Eval examples:    {len(eval_dataset)}")
print(f"Epochs:           {NUM_EPOCHS}")
print(f"Train loss:       {train_result.training_loss:.4f}")
print(f"Eval loss:        {eval_results['eval_loss']:.4f}")
print(f"Training time:    {train_result.metrics['train_runtime']:.1f}s")
print(f"Adapter saved:    {OUTPUT_DIR}/")
print(f"W&B run:          {WANDB_RUN_NAME if USE_WANDB else 'disabled'}")
print(f"{'='*60}")
print(f"\nNext steps:")
print(f"  1. Review W&B dashboard for training curves")
print(f"  2. Test inference on more website descriptions")
print(f"  3. Feed output to AgentQL for browser execution")
print(f"  4. Push adapter to HuggingFace hackathon org")
