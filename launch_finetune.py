"""
============================================================
Agentic World — Launch Fine-Tuning Job via W&B
============================================================
Client-side script that:
  1. Uploads training data (train.jsonl + eval.jsonl) as a W&B artifact
  2. Writes a config JSON with hyperparameters
  3. Submits a fine-tuning job to W&B Launch via the CLI

Prerequisites on the NVIDIA instance:
    pip install wandb unsloth trl transformers peft accelerate bitsandbytes datasets
    wandb login
    wandb launch-agent --queue <queue-name>

Usage:
    # Upload data + launch with defaults
    python launch_finetune.py

    # Custom hyperparams
    python launch_finetune.py --epochs 10 --lora-rank 64 --learning-rate 1e-4 --queue gpu-a10g

    # Just upload data (no launch)
    python launch_finetune.py --upload-only

    # Launch with existing artifact
    python launch_finetune.py --data-artifact agentic-world/training-data:v3

    # Dry run (print what would happen)
    python launch_finetune.py --dry-run
============================================================
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
from pathlib import Path

import wandb

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Upload data & launch fine-tuning via W&B")

# Data
parser.add_argument("--train-file", type=str, default="train.jsonl", help="Path to training JSONL")
parser.add_argument("--eval-file", type=str, default="eval.jsonl", help="Path to eval JSONL")
parser.add_argument("--data-artifact", type=str, default=None,
                    help="Existing W&B artifact (skip upload). e.g. 'entity/agentic-world/training-data:v2'")

# Model
parser.add_argument("--model", type=str, default="unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit")

# LoRA
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=64)
parser.add_argument("--lora-dropout", type=float, default=0.05)

# Training
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--lr-scheduler", type=str, default="cosine")

# Launch
parser.add_argument("--queue", "-q", type=str, default="gpu-a10g",
                    help="W&B Launch queue name (must match your NVIDIA instance agent)")
parser.add_argument("--project", type=str, default="agentic-world")
parser.add_argument("--entity", "-e", type=str, default=None, help="W&B entity (team or username)")
parser.add_argument("--job-name", type=str, default="finetune-behavioral",
                    help="Name for the Launch job")

# Modes
parser.add_argument("--upload-only", action="store_true", help="Just upload data, don't launch")
parser.add_argument("--dry-run", action="store_true", help="Print config but don't execute")
parser.add_argument("--no-queue", action="store_true",
                    help="Run locally instead of queuing to Launch (for testing)")

args = parser.parse_args()

# ============================================================
# UPLOAD TRAINING DATA AS ARTIFACT
# ============================================================

print("=" * 60)
print("AGENTIC WORLD — LAUNCH FINE-TUNING")
print("=" * 60)

data_artifact_name = args.data_artifact

if not data_artifact_name:
    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)

    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        sys.exit(1)
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found")
        sys.exit(1)

    train_count = sum(1 for _ in open(train_path))
    eval_count = sum(1 for _ in open(eval_path))
    train_size = train_path.stat().st_size / 1024
    eval_size = eval_path.stat().st_size / 1024

    print(f"\nData:")
    print(f"  Train: {train_path} ({train_count} examples, {train_size:.1f} KB)")
    print(f"  Eval:  {eval_path} ({eval_count} examples, {eval_size:.1f} KB)")

    if args.dry_run:
        data_artifact_name = "<entity>/agentic-world/training-data:latest"
        print(f"\n[DRY RUN] Would upload -> {data_artifact_name}")
    else:
        print(f"\nUploading data to W&B...")
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            job_type="upload-data",
            name="upload-training-data",
        )

        artifact = wandb.Artifact(
            name="training-data",
            type="dataset",
            description=f"Behavioral fine-tuning data: {train_count} train, {eval_count} eval",
            metadata={
                "train_examples": train_count,
                "eval_examples": eval_count,
            },
        )
        artifact.add_file(str(train_path), name="train.jsonl")
        artifact.add_file(str(eval_path), name="eval.jsonl")

        run.log_artifact(artifact)
        artifact.wait()

        data_artifact_name = f"{run.entity}/{args.project}/training-data:latest"
        print(f"  Uploaded: {data_artifact_name}")
        run.finish()

    if args.upload_only:
        print(f"\nDone. Launch later with:")
        print(f"  python launch_finetune.py --data-artifact {data_artifact_name}")
        sys.exit(0)
else:
    print(f"\nUsing existing artifact: {data_artifact_name}")

# ============================================================
# BUILD LAUNCH CONFIG
# ============================================================

run_config = {
    "overrides": {
        "run_config": {
            "model_name": args.model,
            "max_seq_length": 4096,
            "load_in_4bit": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": 0.01,
            "lr_scheduler": args.lr_scheduler,
            "seed": 42,
            "data_artifact": data_artifact_name,
            "train_file": "train.jsonl",
            "eval_file": "eval.jsonl",
            "output_dir": "outputs/mistral-nemo-behavioral-lora",
        },
    },
}

print(f"\nJob config:")
rc = run_config["overrides"]["run_config"]
for k, v in rc.items():
    if k != "target_modules":
        print(f"  {k}: {v}")

print(f"\nQueue: {args.queue}")

if args.dry_run:
    print(f"\n[DRY RUN] Would run:")
    print(f"  wandb launch --uri . --job-name {args.job_name} "
          f"--queue {args.queue} --project {args.project} "
          f"--entry-point 'python finetune_job.py'")
    print(f"\nConfig JSON:")
    print(json.dumps(run_config, indent=2))
    sys.exit(0)

# ============================================================
# SUBMIT VIA wandb launch CLI
# ============================================================

# Write config to temp file
config_file = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", prefix="wandb_launch_config_", delete=False,
)
json.dump(run_config, config_file, indent=2)
config_file.close()

print(f"\nSubmitting to W&B Launch...")
print(f"  Config: {config_file.name}")

script_dir = str(Path(__file__).parent.resolve())

cmd = [
    "wandb", "launch",
    "--uri", script_dir,
    "--job-name", args.job_name,
    "--entry-point", "python finetune_job.py",
    "--project", args.project,
    "--config", config_file.name,
]

if args.queue and not args.no_queue:
    cmd.extend(["--queue", args.queue])

if args.entity:
    cmd.extend(["--entity", args.entity])

print(f"  Command: {' '.join(cmd)}\n")

result = subprocess.run(cmd, text=True)

# Cleanup temp config
os.unlink(config_file.name)

if result.returncode != 0:
    print(f"\nLaunch failed (exit code {result.returncode})")
    sys.exit(1)

print(f"\n{'='*60}")
print("LAUNCH SUBMITTED")
print("=" * 60)
print(f"  Queue:   {args.queue}")
print(f"  Data:    {data_artifact_name}")
print(f"  Project: {args.project}")
print(f"\n  The job runs when your Launch agent picks it up.")
print(f"  Start the agent on your NVIDIA instance with:")
print(f"    wandb launch-agent --queue {args.queue}")
print(f"\n  Monitor at: https://wandb.ai/{args.entity or '<entity>'}/{args.project}")
print("=" * 60)
