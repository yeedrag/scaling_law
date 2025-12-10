#!/usr/bin/env python3
"""
Light filter for GSM8K: quickly remove problems that can be solved without reasoning.

This is a FAST filter that:
1. Uses only direct answering (no CoT generation needed)
2. Runs multiple direct attempts per question
3. Filters out questions where the model gets it right without reasoning

Use this to further filter an already-filtered dataset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

from openai import AsyncOpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Prompts - Different direct prompting strategies
# =============================================================================

DIRECT_PROMPTS = [
    # Strategy 1: Just answer
    dedent("""
        Answer this math question with ONLY the final numeric answer. No explanation.
        Question: {question}
        Answer:
    """).strip(),
    
    # Strategy 2: Think briefly then answer
    dedent("""
        Quick! What's the answer to this math question? Just the number.
        Question: {question}
        Number:
    """).strip(),
    
    # Strategy 3: Intuitive answer
    dedent("""
        Give me just the number answer to this problem. Trust your intuition.
        {question}
        =
    """).strip(),
]


# =============================================================================
# Helpers
# =============================================================================

def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison."""
    if not s:
        return None
    s = s.strip().replace(',', '').replace('$', '').replace(' ', '')
    s = re.sub(r'\.0+$', '', s)
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def extract_direct_answer(text: str) -> Optional[str]:
    """Extract numeric answer from direct response."""
    text = text.strip()
    match = re.search(r'-?[\d,]+\.?\d*', text.replace(',', ''))
    if match:
        return normalize_number(match.group())
    return None


async def call_openai_async(client: AsyncOpenAI, model: str, prompt: str) -> str:
    """Async OpenAI call."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=2048,
    )
    return resp.choices[0].message.content or ""


async def judge_correctness(client: AsyncOpenAI, model: str, reference: str, candidate: str) -> bool:
    """Judge if two answers match."""
    if not candidate:
        return False
    
    ref_norm = normalize_number(reference)
    cand_norm = normalize_number(candidate)
    if ref_norm and cand_norm and ref_norm == cand_norm:
        return True
    
    prompt = f"Are '{reference}' and '{candidate}' the same number? Reply only 'yes' or 'no'."
    response = await call_openai_async(client, model, prompt)
    return response.strip().lower().startswith("yes")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Light filter: remove easy problems")
    parser.add_argument("--input", required=True, help="Input JSON file (filtered_gsm8k.questions.json)")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--runs", type=int, default=3, help="Direct attempts per question")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output file (default: input_lightfiltered.json)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vLLM")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required")

    # Load input
    print(f"Loading: {args.input}")
    with open(args.input) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems")

    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.with_suffix("")) + "_lightfiltered.json"

    # Init vLLM
    print(f"\nLoading model: {args.model}")
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "max_num_seqs": args.batch_size,
    }
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling = SamplingParams(temperature=1.0, max_tokens=32, top_p=1.0)

    async_client = AsyncOpenAI()

    # ==========================================================================
    # Generate direct answers
    # ==========================================================================
    print(f"\nGenerating {args.runs} direct attempts per question using {len(DIRECT_PROMPTS)} prompt strategies...")
    
    # Create all prompts
    all_prompts = []
    prompt_mapping = []  # (problem_idx, run_idx, prompt_strategy_idx)
    
    for p_idx, p in enumerate(problems):
        for run_idx in range(args.runs):
            strategy_idx = run_idx % len(DIRECT_PROMPTS)
            prompt_template = DIRECT_PROMPTS[strategy_idx]
            msg = prompt_template.format(question=p["question"])
            messages = [{"role": "user", "content": msg}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompts.append(formatted)
            prompt_mapping.append((p_idx, run_idx, strategy_idx))
    
    print(f"  Total prompts: {len(all_prompts)}")
    outputs = llm.generate(all_prompts, sampling)
    
    # Extract answers
    print("  Extracting answers...")
    extracted = [extract_direct_answer(out.outputs[0].text.strip()) for out in outputs]
    
    # ==========================================================================
    # Judge all answers
    # ==========================================================================
    print("Judging answers...")
    
    async def judge_all():
        tasks = []
        for (p_idx, _, _), ans in zip(prompt_mapping, extracted):
            expected = problems[p_idx]["answer"]
            tasks.append(judge_correctness(async_client, args.judge_model, expected, ans))
        
        # Process in chunks to avoid rate limits
        CHUNK = 100
        results = []
        for i in range(0, len(tasks), CHUNK):
            chunk = tasks[i:i+CHUNK]
            print(f"  Judging {i+1}-{min(i+len(chunk), len(tasks))}/{len(tasks)}...")
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            if i + CHUNK < len(tasks):
                await asyncio.sleep(0.5)
        return results
    
    judgments = asyncio.run(judge_all())
    
    # ==========================================================================
    # Analyze results
    # ==========================================================================
    print("\nAnalyzing results...")
    
    # Group by problem
    problem_results = {i: [] for i in range(len(problems))}
    for (p_idx, run_idx, _), correct in zip(prompt_mapping, judgments):
        problem_results[p_idx].append(correct)
    
    # Categorize
    kept = []  # Problems where ALL direct attempts failed
    removed = []  # Problems where ANY direct attempt succeeded
    
    for p_idx, p in enumerate(problems):
        any_correct = any(problem_results[p_idx])
        success_rate = sum(problem_results[p_idx]) / len(problem_results[p_idx])
        
        result = {
            **p,
            "direct_success_rate": success_rate,
            "direct_any_correct": any_correct,
        }
        
        if any_correct:
            removed.append(result)
        else:
            kept.append(result)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Input:   {len(problems)} problems")
    print(f"  Kept:    {len(kept)} ({100*len(kept)/len(problems):.1f}%) - require reasoning")
    print(f"  Removed: {len(removed)} ({100*len(removed)/len(problems):.1f}%) - solvable without reasoning")
    
    # Save
    output_data = {
        "source": args.input,
        "model": args.model,
        "runs": args.runs,
        "summary": {
            "input": len(problems),
            "kept": len(kept),
            "removed": len(removed),
        },
        "problems": kept,
        "removed": removed,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {args.output}")
    
    # Also save simple list
    simple_path = Path(args.output).with_suffix(".questions.json")
    simple_data = [{"question": p["question"], "answer": p["answer"]} for p in kept]
    with open(simple_path, "w") as f:
        json.dump(simple_data, f, indent=2)
    print(f"Simple list: {simple_path}")


if __name__ == "__main__":
    main()

