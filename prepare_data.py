"""
============================================================
Agentic World — Data Preparation
============================================================
Converts raw behavioral description .txt files (from PostHog
session analysis) into fine-tuning JSONL format.

Input:  Directory of .txt files (behavioral descriptions)
Output: train.jsonl + eval.jsonl

Usage:
    python prepare_data.py --input descriptions/ --train train.jsonl --eval eval.jsonl --eval-split 7
============================================================
"""

import os
import json
import random
import argparse

# FunCity site description — constant input for all training examples
SITE_DESCRIPTION = """Website: https://fun-city-xi.vercel.app/
Description: FunCity is a Reddit-style NYC discovery board where users browse posts organized by NYC boroughs (The Bronx, Brooklyn, Manhattan, Queens, Staten Island) and topics (Art & Culture, Food & Eats, Hidden Gems, Nature & Parks, Nightlife). The homepage shows a feed of user posts sorted by Hot, New, or Top tabs. Each post card displays a borough tag, username, timestamp, title, body preview, upvote/downvote arrows with score, and comment count. The right sidebar contains borough filter buttons, topic filter buttons, and a Trending section showing top 5 posts. Users can click posts to see the full post detail page with a comments thread (each comment has upvote/downvote). There is a Sign Up button (top right) with a modal collecting username, password, age group, country, and NYC familiarity. Logged-in users see a "+ New Post" button and can comment and vote."""

SYSTEM_PROMPT = """You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."""


def clean_description(text: str) -> str:
    """Strip markdown backtick wrappers from descriptions."""
    text = text.strip()
    if text.startswith('```'):
        text = text[3:]
        if text.startswith('\n'):
            text = text[1:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def make_example(description: str) -> dict:
    """Create a chat-completion training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SITE_DESCRIPTION},
            {"role": "assistant", "content": description},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare behavioral data for fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Directory containing .txt description files")
    parser.add_argument("--train", type=str, default="train.jsonl", help="Output training JSONL file")
    parser.add_argument("--eval", type=str, default="eval.jsonl", help="Output eval JSONL file")
    parser.add_argument("--eval-split", type=int, default=7, help="Number of examples to hold out for eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split")
    args = parser.parse_args()

    # Find all .txt files
    txt_files = sorted([
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith('.txt')
    ])

    print(f"Found {len(txt_files)} description files in {args.input}/")

    if len(txt_files) == 0:
        print("ERROR: No .txt files found!")
        return

    # Load and clean all descriptions
    examples = []
    for filepath in txt_files:
        with open(filepath) as f:
            raw = f.read()
        cleaned = clean_description(raw)
        if len(cleaned) < 100:
            print(f"  WARNING: Skipping {os.path.basename(filepath)} (too short: {len(cleaned)} chars)")
            continue
        examples.append(make_example(cleaned))
        print(f"  OK: {os.path.basename(filepath)}: {len(cleaned)} chars")

    print(f"\nTotal valid examples: {len(examples)}")

    # Train/eval split
    random.seed(args.seed)
    indices = list(range(len(examples)))
    random.shuffle(indices)

    eval_count = min(args.eval_split, len(examples) // 4)  # Never more than 25%
    eval_indices = set(indices[:eval_count])

    train_examples = [ex for i, ex in enumerate(examples) if i not in eval_indices]
    eval_examples = [ex for i, ex in enumerate(examples) if i in eval_indices]

    # Write JSONL files
    with open(args.train, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(args.eval, 'w') as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + '\n')

    # Stats
    train_chars = sum(len(ex["messages"][2]["content"]) for ex in train_examples)
    eval_chars = sum(len(ex["messages"][2]["content"]) for ex in eval_examples)

    print(f"\n{'='*50}")
    print(f"DATASET PREPARED")
    print(f"{'='*50}")
    print(f"Training:   {len(train_examples)} examples ({train_chars:,} chars, ~{train_chars//4:,} tokens)")
    print(f"Evaluation: {len(eval_examples)} examples ({eval_chars:,} chars, ~{eval_chars//4:,} tokens)")
    print(f"Output:     {args.train}, {args.eval}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
