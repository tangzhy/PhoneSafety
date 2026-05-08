#!/usr/bin/env python3
"""
PhoneSafety: Run inference on the 700 safety-critical moments.

This script loads the benchmark, constructs multimodal prompts (text + screenshot)
with the safety protocol, and queries a model via OpenAI-compatible API (vLLM, SGLang, etc.).

Usage:
    python inference/run_inference.py \
        --api_base http://localhost:8100/v1 \
        --api_key token-placeholder \
        --model_name /path/to/your-model \
        --protocol strict \
        --output_file outputs/your_model_strict.jsonl

Requirements:
    pip install openai
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

from protocols import get_protocol_prompt
from model_prompts import get_model_system_prompt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BENCHMARK = DATA_DIR / "phonesafety_700.jsonl"


def load_benchmark(path: Path) -> list:
    """Load benchmark cases from JSONL."""
    cases = []
    with open(path) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def get_screenshot_path(case: dict) -> Path | None:
    """Resolve the screenshot path for a case."""
    case_id = case["case_id"]
    screenshots_dir = DATA_DIR / "screenshots"
    for ext in ("jpg", "png", "jpeg"):
        p = screenshots_dir / f"{case_id}.{ext}"
        if p.exists():
            return p
    return None


def build_prompt(case: dict, system_msg: str) -> list:
    """Build the multimodal message list for one case (text + screenshot)."""

    history = case.get("action_history", "")
    history_section = f"Action history:\n{history}" if history else "Action history: (none)"

    user_text = (
        f"User instruction: {case['instruction']}\n\n"
        f"{history_section}\n\n"
        f"Current screenshot is provided. Please predict the next phone action."
    )

    # Build multimodal content with screenshot
    img_path = get_screenshot_path(case)
    if img_path:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        ext = img_path.suffix[1:]
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": user_text},
        ]
    else:
        # Fallback: text-only if screenshot not found
        user_content = user_text

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    return messages


def run_inference(args):
    """Run inference on all benchmark cases."""
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    protocol_text = get_protocol_prompt(args.protocol, model_variant=args.format)
    system_msg = get_model_system_prompt(args.format, privacy_prompt=protocol_text)
    cases = load_benchmark(Path(args.benchmark))

    print(f"Loaded {len(cases)} cases")
    print(f"Protocol: {args.protocol}")
    print(f"Format: {args.format}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_file}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output
    existing_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_ids.add(d.get("case_id", ""))
        print(f"Resuming: {len(existing_ids)} cases already done")

    with open(output_path, "a") as out_f:
        for i, case in enumerate(cases):
            if case["case_id"] in existing_ids:
                continue

            messages = build_prompt(case, system_msg)

            try:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                pred_text = response.choices[0].message.content
            except Exception as e:
                pred_text = f"ERROR: {str(e)}"

            result = {
                "case_id": case["case_id"],
                "violation_type": case["violation_type"],
                "instruction": case["instruction"],
                "pred_response": pred_text,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(cases)}")

    print(f"Done. Output saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="PhoneSafety inference")
    parser.add_argument("--api_base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", default="token-placeholder", help="API key")
    parser.add_argument("--model_name", required=True, help="Model name/path")
    parser.add_argument("--protocol", default="strict", choices=["strict", "minimal"],
                        help="Safety protocol to use")
    parser.add_argument("--format", default="standard",
                        choices=["standard", "claude", "gemini", "seed", "kimi",
                                 "autoglm", "gelab", "mobile_agent", "mai_ui"],
                        help="Model output format (determines system prompt and action format)")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK),
                        help="Path to benchmark JSONL")
    parser.add_argument("--output_file", required=True, help="Output JSONL path")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
