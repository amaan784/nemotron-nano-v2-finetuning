"""
============================================================
Agentic World — Inference Pipeline
============================================================
Loads fine-tuned Mistral Nemo 12B LoRA adapter and generates
behavioral profiles for any website description.

Usage:
    # Single website
    python inference.py --url https://example.com --description "An e-commerce site..."

    # Interactive mode
    python inference.py --interactive

    # Batch mode
    python inference.py --batch sites.json

    # Custom adapter path
    python inference.py --adapter ./my-adapter --url ...

Output: Behavioral profile + AgentQL-compatible action plan
============================================================
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "outputs/mistral-nemo-behavioral-lora"
MAX_SEQ_LENGTH = 4096

SYSTEM_PROMPT = """You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."""

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Generate behavioral profiles for websites")
parser.add_argument("--url", type=str, help="Website URL")
parser.add_argument("--description", type=str, help="Website description")
parser.add_argument("--interactive", action="store_true", help="Interactive mode")
parser.add_argument("--batch", type=str, help="Path to JSON file with multiple sites")
parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER, help="Path to LoRA adapter")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--output", type=str, help="Output file path (JSON)")
args = parser.parse_args()

# ============================================================
# LOAD MODEL + ADAPTER
# ============================================================

print("=" * 60)
print("AGENTIC WORLD — BEHAVIORAL INFERENCE")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print(f"\n📦 Loading base model: {args.model}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter if it exists
if os.path.exists(args.adapter):
    print(f"🔧 Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print(f"🔀 Merging adapter into base model for fast inference...")
    model = model.merge_and_unload()
    print(f"✅ Adapter merged")
else:
    print(f"⚠️  No adapter found at {args.adapter}, using base model")

model.eval()
print(f"✅ Model ready for inference\n")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_profile(url: str, description: str) -> str:
    """Generate a behavioral profile for a website."""
    user_content = f"Website: {url}\nDescription: {description}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


def profile_to_action_plan(profile: str, url: str) -> dict:
    """Convert behavioral profile to AgentQL-compatible action plan."""
    return {
        "url": url,
        "behavioral_profile": profile,
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "adapter": args.adapter if os.path.exists(args.adapter) else None,
        "agentql_prompt": (
            f"You are a browser automation agent. Visit {url} and interact with it "
            f"exactly as described in this behavioral profile:\n\n{profile}\n\n"
            f"Execute each action in sequence with the specified timings. "
            f"Use AgentQL queries to locate elements and interact with them."
        ),
    }

# ============================================================
# EXECUTION MODES
# ============================================================

if args.interactive:
    # Interactive mode
    print("🎮 Interactive mode — type 'quit' to exit\n")
    results = []

    while True:
        url = input("URL: ").strip()
        if url.lower() == "quit":
            break

        description = input("Description: ").strip()
        if description.lower() == "quit":
            break

        print(f"\n⏳ Generating behavioral profile...")
        profile = generate_profile(url, description)

        print(f"\n{'='*60}")
        print(f"BEHAVIORAL PROFILE — {url}")
        print(f"{'='*60}")
        print(profile)
        print(f"\n{'='*60}")
        print(f"Length: {len(profile)} chars, ~{len(profile)//4} tokens\n")

        results.append(profile_to_action_plan(profile, url))

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved {len(results)} profiles to {args.output}")

elif args.batch:
    # Batch mode
    print(f"📦 Batch mode — loading {args.batch}")
    with open(args.batch) as f:
        sites = json.load(f)

    results = []
    for i, site in enumerate(sites):
        url = site["url"]
        description = site["description"]
        print(f"\n[{i+1}/{len(sites)}] Generating profile for {url}...")

        profile = generate_profile(url, description)
        result = profile_to_action_plan(profile, url)
        results.append(result)

        print(f"   ✅ {len(profile)} chars generated")

    output_path = args.output or f"behavioral_profiles_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved {len(results)} profiles to {output_path}")

elif args.url and args.description:
    # Single-shot mode
    print(f"⏳ Generating behavioral profile for {args.url}...")
    profile = generate_profile(args.url, args.description)

    print(f"\n{'='*60}")
    print(f"BEHAVIORAL PROFILE — {args.url}")
    print(f"{'='*60}")
    print(profile)
    print(f"\n{'='*60}")
    print(f"Length: {len(profile)} chars, ~{len(profile)//4} tokens")

    if args.output:
        result = profile_to_action_plan(profile, args.url)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Saved to {args.output}")

else:
    print("Usage:")
    print("  python inference.py --url URL --description 'Site description...'")
    print("  python inference.py --interactive")
    print("  python inference.py --batch sites.json")
    sys.exit(1)
