"""
============================================================
Agentic World — Data Preparation
============================================================
Converts raw behavioral description .txt files into JSONL
format for fine-tuning with Mistral Nemo 12B.

Input:  Directory of .txt files (one behavioral profile each)
Output: train.jsonl + eval.jsonl in chat completion format

Each example becomes:
    system: behavioral simulation prompt
    user:   website URL + description
    assistant: behavioral profile

Usage:
    python prepare_data.py --input ./descriptions --output ./ --eval-count 7
============================================================
"""

import os
import json
import glob
import random
import argparse

# ============================================================
# CONFIG
# ============================================================

SYSTEM_PROMPT = """You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."""

# FunCity site description (constant for all training examples)
SITE_DESCRIPTION = """Website: https://fun-city-xi.vercel.app/
Description: FunCity is a Reddit-style NYC discovery board where users browse posts organized by NYC boroughs (The Bronx, Brooklyn, Manhattan, Queens, Staten Island) and topics (Art & Culture, Food & Eats, Hidden Gems, Nature & Parks, Nightlife). The homepage shows a feed of user posts sorted by Hot, New, or Top tabs. Each post card displays a borough tag, username, timestamp, title, body preview, upvote/downvote arrows with score, and comment count. The right sidebar contains borough filter buttons, topic filter buttons, and a Trending section showing top 5 posts. Users can click posts to see the full post detail page with a comments thread (each comment has upvote/downvote). There is a Sign Up button (top right) with a modal collecting username, password, age group, country, and NYC familiarity. Logged-in users see a "+ New Post" button and can comment and vote."""

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Prepare fine-tuning data from behavioral descriptions")
parser.add_argument("--input", type=str, default="./descriptions", help="Directory of .txt files")
parser.add_argument("--output", type=str, default="./", help="Output directory for JSONL files")
parser.add_argument("--eval-count", type=int, default=7, help="Number of eval examples")
parser.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split")
args = parser.parse_args()

# ============================================================
# LOAD AND PROCESS FILES
# ============================================================

print(f"📂 Loading .txt files from {args.input}")

txt_files = sorted(glob.glob(os.path.join(args.input, "*.txt")))
if not txt_files:
    print(f"❌ No .txt files found in {args.input}")
    exit(1)

examples = []
for fpath in txt_files:
    with open(fpath, "r") as f:
        content = f.read().strip()

    # Strip markdown code block wrappers if present
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```markdown or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines).strip()

    if len(content) < 100:
        print(f"   ⚠️  Skipping {os.path.basename(fpath)} (too short: {len(content)} chars)")
        continue

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SITE_DESCRIPTION},
            {"role": "assistant", "content": content},
        ]
    }
    examples.append(example)
    print(f"   ✅ {os.path.basename(fpath)} — {len(content)} chars")

print(f"\n📊 Total examples: {len(examples)}")

# ============================================================
# TRAIN/EVAL SPLIT
# ============================================================

random.seed(args.seed)
random.shuffle(examples)

eval_count = min(args.eval_count, len(examples) // 4)  # Max 25% for eval
eval_examples = examples[:eval_count]
train_examples = examples[eval_count:]

print(f"   Train: {len(train_examples)}")
print(f"   Eval:  {len(eval_examples)}")

# ============================================================
# WRITE JSONL
# ============================================================

train_path = os.path.join(args.output, "train.jsonl")
eval_path = os.path.join(args.output, "eval.jsonl")

with open(train_path, "w") as f:
    for ex in train_examples:
        f.write(json.dumps(ex) + "\n")

with open(eval_path, "w") as f:
    for ex in eval_examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n💾 Saved:")
print(f"   {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
print(f"   {eval_path} ({os.path.getsize(eval_path) / 1024:.1f} KB)")
print(f"\nReady for: python finetune.py")
