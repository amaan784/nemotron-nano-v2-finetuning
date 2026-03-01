"""
============================================================
Agentic World — Fine-Tuning Pipeline
============================================================
Fine-tunes Mistral Nemo 12B Instruct on behavioral session
descriptions using Unsloth + QLoRA.

Track: W&B Fine-Tuning Track (Mistral Worldwide Hackathon)
Model: mistralai/Mistral-Nemo-Instruct-2407
Method: QLoRA (4-bit quantization + LoRA adapters)
Framework: Unsloth + HuggingFace TRL
Tracking: Weights & Biases (Models + Weave)

Why Mistral Nemo 12B:
    - Standard Transformer architecture (no Mamba/hybrid issues)
    - First-class Unsloth support, battle-tested QLoRA
    - 12B params fits comfortably on A10G (24GB) in 4-bit (~7GB)
    - Strong instruction-following out of the box
    - Mistral model = aligned with W&B Fine-Tuning Track rules
    - 128K context window (we use 4096 for training efficiency)

Requirements (install via setup_brev.sh):
    pip install unsloth wandb datasets trl

Usage:
    python finetune.py

    # Override model (fallback to 7B):
    MODEL=unsloth/mistral-7b-instruct-v0.3-bnb-4bit python finetune.py

GPU: A10G (24GB) or A100 (40GB)
Expected VRAM: ~8-12GB with QLoRA
Expected time: 10-20 minutes for 37 examples
============================================================
"""

import os
import json
import torch
from datetime import datetime
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ============================================================
# CONFIG
# ============================================================

# Model — override with env var for quick fallback
MODEL_NAME = os.environ.get(
    "MODEL",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
)
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = True

# LoRA — all standard Transformer linear projections
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
SEED = 42

# Data
TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"

# Output
OUTPUT_DIR = "outputs/mistral-nemo-behavioral-lora"
HF_REPO = os.environ.get("HF_REPO", None)

# W&B
WANDB_PROJECT = "agentic-world"
WANDB_RUN_NAME = f"mistral-nemo-behavioral-{datetime.now().strftime('%Y%m%d-%H%M')}"

# ============================================================
# SETUP W&B
# ============================================================

print("=" * 60)
print("AGENTIC WORLD — BEHAVIORAL MODEL FINE-TUNING")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Track: W&B Fine-Tuning (Mistral Worldwide Hackathon)")
print()

try:
    import wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "lr_scheduler": LR_SCHEDULER,
            "max_seq_length": MAX_SEQ_LENGTH,
            "method": "QLoRA-4bit",
            "weight_decay": WEIGHT_DECAY,
        },
        tags=["hackathon", "mistral-worldwide", "w&b-finetuning-track", "behavioral-finetuning", "mistral-nemo"],
    )
    USE_WANDB = True
    print(f"✅ W&B initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
except Exception as e:
    print(f"⚠️  W&B not available ({e}), continuing without tracking")
    USE_WANDB = False

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\n📦 Loading model: {MODEL_NAME}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}")
print(f"   4-bit quantization: {LOAD_IN_4BIT}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

print(f"✅ Model loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Architecture: Standard Transformer (all layers are attention + MLP)")

# ============================================================
# APPLY LORA
# ============================================================

print(f"\n🔧 Applying LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")
print(f"   Target modules: {TARGET_MODULES}")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA applied")
print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   All {len(TARGET_MODULES)} projection types adapted across every layer")

if USE_WANDB:
    wandb.log({
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
    })

# ============================================================
# LOAD DATASET
# ============================================================

print(f"\n📊 Loading dataset")
print(f"   Train: {TRAIN_FILE}")
print(f"   Eval:  {EVAL_FILE}")

from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "eval": EVAL_FILE,
})

print(f"   Train examples: {len(dataset['train'])}")
print(f"   Eval examples:  {len(dataset['eval'])}")

def formatting_func(examples):
    """Convert messages to Mistral Nemo chat format."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

train_dataset = dataset["train"].map(formatting_func, batched=True, remove_columns=["messages"])
eval_dataset = dataset["eval"].map(formatting_func, batched=True, remove_columns=["messages"])

# Log data stats
text_lengths = [len(t) for t in train_dataset["text"]]
avg_len = sum(text_lengths) / len(text_lengths)
max_len = max(text_lengths)
min_len = min(text_lengths)

print(f"\n📝 Dataset statistics:")
print(f"   Avg text length: {avg_len:.0f} chars")
print(f"   Min / Max: {min_len} / {max_len} chars")
print(f"\n   Sample (first 300 chars):")
print(f"   {train_dataset[0]['text'][:300]}...")

if USE_WANDB:
    wandb.log({
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "avg_text_length": avg_len,
        "max_text_length": max_len,
        "min_text_length": min_len,
    })

# ============================================================
# TRAINING
# ============================================================

print(f"\n🚀 Starting training")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Scheduler: {LR_SCHEDULER}")


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
        packing=False,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
reserved_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
print(f"\n   GPU: {gpu_stats.name}")
print(f"   Total VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")
print(f"   Reserved: {reserved_memory:.1f} GB")

train_result = trainer.train()

peak_memory = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
print(f"\n✅ Training complete!")
print(f"   Training loss: {train_result.training_loss:.4f}")
print(f"   Training time: {train_result.metrics['train_runtime']:.1f}s")
print(f"   Peak VRAM: {peak_memory:.1f} GB")

# ============================================================
# EVALUATION
# ============================================================

print(f"\n📊 Running evaluation...")

eval_results = trainer.evaluate()
print(f"   Eval loss: {eval_results['eval_loss']:.4f}")

if USE_WANDB:
    wandb.log({
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
        "peak_vram_gb": peak_memory,
    })

# ============================================================
# SAVE ADAPTER
# ============================================================

print(f"\n💾 Saving LoRA adapter to {OUTPUT_DIR}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

config_path = os.path.join(OUTPUT_DIR, "training_config.json")
with open(config_path, "w") as f:
    json.dump({
        "base_model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler": LR_SCHEDULER,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": eval_results["eval_loss"],
        "training_time_s": train_result.metrics["train_runtime"],
        "peak_vram_gb": peak_memory,
        "hackathon_track": "W&B Fine-Tuning Track",
    }, f, indent=2)

# Check adapter file size
for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        print(f"✅ Adapter saved ({os.path.getsize(fpath) / 1024 / 1024:.1f} MB)")
        break
else:
    print(f"✅ Adapter saved to {OUTPUT_DIR}/")

# ============================================================
# PUSH TO HUGGING FACE
# ============================================================

if HF_REPO:
    print(f"\n📤 Pushing to Hugging Face: {HF_REPO}")
    model.push_to_hub(HF_REPO, tokenizer=tokenizer)
    print(f"✅ Pushed to {HF_REPO}")
else:
    print(f"\n💡 Push to HF: HF_REPO=mistral-hackaton-2026/agentic-world-lora python finetune.py")

# ============================================================
# LOG W&B ARTIFACT
# ============================================================

if USE_WANDB:
    print(f"\n📦 Logging adapter as W&B artifact...")
    artifact = wandb.Artifact(
        name="mistral-nemo-behavioral-lora",
        type="model",
        description="QLoRA adapter for behavioral simulation — Mistral Nemo 12B fine-tuned on PostHog-derived user behavior profiles",
        metadata={
            "base_model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_results["eval_loss"],
            "train_examples": len(train_dataset),
            "hackathon": "mistral-worldwide-2026",
            "track": "w&b-finetuning",
        },
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
    print(f"✅ Artifact logged to W&B")

# ============================================================
# TEST INFERENCE
# ============================================================

print(f"\n🧪 Testing inference with fine-tuned model...")

FastLanguageModel.for_inference(model)

# Test 1: NEW website — proves generalization
test_input = """Website: https://vinyl-vault.example.com/
Description: An online store selling vintage vinyl records. The homepage features a hero banner with staff picks, a genre filter sidebar (Jazz, Rock, Electronic, Classical, Hip-Hop), and a grid of album cards showing cover art, artist name, album title, price, and condition rating (Mint/VG+/VG/Good). Users can click albums for detail pages with tracklists, seller reviews, and an Add to Cart button. There's a search bar in the top navigation, a wishlist feature, and a cart icon showing item count. New arrivals section at the bottom of the homepage."""

messages = [
    {
        "role": "system",
        "content": "You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."
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
print(f"INFERENCE TEST — New Website (Vinyl Record Store)")
print(f"{'='*60}")
print(response[:2000])
print(f"\nResponse length: {len(response)} chars, ~{len(response)//4} tokens")

if USE_WANDB:
    wandb.log({"test_new_site_response": response, "test_new_site_length": len(response)})

# Test 2: FunCity (training domain) — sanity check
print(f"\n🧪 Sanity check — FunCity (training domain)...")

funcity_input = """Website: https://fun-city-xi.vercel.app/
Description: FunCity is a Reddit-style NYC discovery board where users browse posts organized by NYC boroughs (The Bronx, Brooklyn, Manhattan, Queens, Staten Island) and topics (Art & Culture, Food & Eats, Hidden Gems, Nature & Parks, Nightlife). The homepage shows a feed of user posts sorted by Hot, New, or Top tabs. Each post card displays a borough tag, username, timestamp, title, body preview, upvote/downvote arrows with score, and comment count. The right sidebar contains borough filter buttons, topic filter buttons, and a Trending section showing top 5 posts. Users can click posts to see the full post detail page with a comments thread (each comment has upvote/downvote). There is a Sign Up button (top right) with a modal collecting username, password, age group, country, and NYC familiarity. Logged-in users see a "+ New Post" button and can comment and vote."""

messages_fc = [
    {
        "role": "system",
        "content": "You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."
    },
    {"role": "user", "content": funcity_input},
]

input_ids_fc = tokenizer.apply_chat_template(
    messages_fc, tokenize=True, add_generation_prompt=True, return_tensors="pt",
).to("cuda")

output_fc = model.generate(
    input_ids=input_ids_fc, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True,
)
response_fc = tokenizer.decode(output_fc[0][input_ids_fc.shape[1]:], skip_special_tokens=True)

print(f"\n{'='*60}")
print(f"SANITY CHECK — FunCity")
print(f"{'='*60}")
print(response_fc[:2000])

if USE_WANDB:
    wandb.log({"test_funcity_response": response_fc, "test_funcity_length": len(response_fc)})
    wandb.finish()

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE — SUMMARY")
print(f"{'='*60}")
print(f"Model:            {MODEL_NAME}")
print(f"Architecture:     Standard Transformer (Mistral Nemo 12B)")
print(f"Method:           QLoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
print(f"Target modules:   {', '.join(TARGET_MODULES)}")
print(f"Train examples:   {len(train_dataset)}")
print(f"Eval examples:    {len(eval_dataset)}")
print(f"Epochs:           {NUM_EPOCHS}")
print(f"Train loss:       {train_result.training_loss:.4f}")
print(f"Eval loss:        {eval_results['eval_loss']:.4f}")
print(f"Training time:    {train_result.metrics['train_runtime']:.1f}s")
print(f"Peak VRAM:        {peak_memory:.1f} GB")
print(f"Adapter saved:    {OUTPUT_DIR}/")
print(f"W&B run:          {WANDB_RUN_NAME if USE_WANDB else 'disabled'}")
print(f"{'='*60}")
print(f"\nNext steps:")
print(f"  1. Review W&B dashboard for training curves")
print(f"  2. Run: python inference.py --url https://example.com --description '...'")
print(f"  3. Push adapter: HF_REPO=mistral-hackaton-2026/agentic-world-lora python finetune.py")
print(f"  4. Deploy on Brev for NVIDIA on-device track")
print(f"  5. Add W&B Weave tracing to agent pipeline")
