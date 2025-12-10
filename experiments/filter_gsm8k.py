#!/usr/bin/env python3
"""
Filter GSM8K questions to find ones that require reasoning.

Keeps questions where:
- Model gets it WRONG without reasoning (direct answer)
- Model gets it RIGHT with reasoning (CoT)

Filters out:
- Questions model gets right without reasoning (too easy, no reasoning needed)
- Questions model can't solve even with reasoning (too hard or ambiguous)
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

from src.dataset import load_problems


# =============================================================================
# Prompts
# =============================================================================

# Prompt for direct answer (no reasoning)
DIRECT_PROMPT = dedent("""
    Answer this math question with ONLY the final numeric answer. No explanation, no steps.
    Just output the number.
    
    Question: {question}
    
    Answer:
""").strip()

# Prompt for reasoning (CoT) - model will naturally reason
REASONING_PROMPT = dedent("""
    Solve this math problem step by step. Think carefully through each step, then provide your final numeric answer.
    
    Question: {question}
""").strip()


# =============================================================================
# Helpers
# =============================================================================

def parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_number(s: str) -> Optional[str]:
    """Normalize a number string for comparison."""
    if not s:
        return None
    # Remove common formatting: $, commas, spaces, leading zeros
    s = s.strip().replace(',', '').replace('$', '').replace(' ', '')
    # Remove trailing .0 or .00 etc
    s = re.sub(r'\.0+$', '', s)
    # Try to parse as float and back to handle edge cases
    try:
        val = float(s)
        # Return as int if it's a whole number
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from model response using regex patterns.
    Tries multiple patterns in order of specificity.
    """
    if not text:
        return None
    
    # Pattern 1: \boxed{answer} - most reliable for DeepSeek-R1
    boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed_matches:
        # Take the last boxed answer (final answer)
        answer = boxed_matches[-1].strip()
        # Extract just the number
        num_match = re.search(r'-?[\d,]+\.?\d*', answer.replace(',', ''))
        if num_match:
            return normalize_number(num_match.group())
    
    # Pattern 2: <answer>X</answer>
    answer_tag = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_tag:
        answer = answer_tag[-1].strip()
        num_match = re.search(r'-?[\d,]+\.?\d*', answer.replace(',', ''))
        if num_match:
            return normalize_number(num_match.group())
    
    # Pattern 3: #### answer (GSM8K format)
    hash_match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if hash_match:
        return normalize_number(hash_match.group(1))
    
    # Pattern 4: "The answer is X" or "the final answer is X"
    answer_is = re.findall(r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+\$?(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if answer_is:
        return normalize_number(answer_is[-1])
    
    # Pattern 5: "= X" at the end of reasoning, looking for the last equation result
    # Look in the last 500 chars for the final calculation
    last_part = text[-500:] if len(text) > 500 else text
    equals_matches = re.findall(r'=\s*\$?(-?[\d,]+\.?\d*)\s*(?:dollars?|miles?|hours?|minutes?|people?|items?|books?|apples?|\.|\s|$)', last_part, re.IGNORECASE)
    if equals_matches:
        return normalize_number(equals_matches[-1])
    
    # Pattern 6: Bold or emphasized numbers **X** in the last part
    bold_matches = re.findall(r'\*\*(-?[\d,]+\.?\d*)\*\*', last_part)
    if bold_matches:
        return normalize_number(bold_matches[-1])
    
    # Pattern 7: Last number in the response (fallback)
    all_numbers = re.findall(r'-?[\d,]+\.?\d*', text.replace(',', ''))
    if all_numbers:
        # Filter out very small numbers that are likely indices or examples
        substantial = [n for n in all_numbers if len(n.replace('.', '').replace('-', '')) > 0]
        if substantial:
            return normalize_number(substantial[-1])
    
    return None


async def call_openai_async(client: AsyncOpenAI, model: str, prompt: str, temperature: float = 1.0) -> str:
    """Async OpenAI chat completion."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=1024,
    )
    return resp.choices[0].message.content or ""


async def judge_correctness_async(
    client: AsyncOpenAI,
    model: str,
    reference: str,
    candidate: str,
) -> bool:
    """
    Use LLM to judge if two answers are equivalent.
    Only compares numbers, no extraction needed.
    """
    if not candidate:
        return False
    
    # First try direct comparison after normalization
    ref_norm = normalize_number(reference)
    cand_norm = normalize_number(candidate)
    if ref_norm and cand_norm and ref_norm == cand_norm:
        return True
    
    # If direct match fails, use LLM to check equivalence
    prompt = dedent(f"""
        Are these two numeric answers equivalent? Consider:
        - Different formatting (50 vs 50.0 vs $50)
        - Rounding (3.33 vs 3.333)
        - Units don't matter, just the number
        
        Reference: {reference}
        Candidate: {candidate}
        
        Reply with ONLY "yes" or "no".
    """).strip()
    
    response = await call_openai_async(client, model, prompt)
    return response.strip().lower().startswith("yes")


def extract_direct_answer(text: str) -> Optional[str]:
    """Extract numeric answer from direct response (should be just a number)."""
    text = text.strip()
    # For direct answers, just find the first number
    match = re.search(r'-?[\d,]+\.?\d*', text.replace(',', ''))
    if match:
        return normalize_number(match.group())
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Filter GSM8K for reasoning-necessary questions")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--limit", type=int, default=100, help="Number of questions to process")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--output", default="filtered_gsm8k.json", help="Output file path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM generation")
    parser.add_argument("--direct-runs", type=int, default=5, help="Number of times to run direct mode (no reasoning) per question")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required for judging answers")

    print(f"Loading GSM8K ({args.split} split, limit={args.limit})...")
    problems = load_problems("openai/gsm8k", args.split, subset="main", limit=args.limit, shuffle_seed=42)
    if not problems:
        raise RuntimeError("No problems loaded")
    print(f"Loaded {len(problems)} problems")

    # Initialize vLLM model
    print(f"\nLoading model: {args.model}")
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": 4096,
        "max_num_seqs": args.batch_size,
    }
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Sampling params
    direct_sampling = SamplingParams(temperature=1, max_tokens=10, top_p=1.0)
    reasoning_sampling = SamplingParams(temperature=1, max_tokens=2048, top_p=0.9)

    async_client = AsyncOpenAI()

    # ==========================================================================
    # Combined generation + judging pipeline (process in parallel)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Running generation and judging in parallel")
    print("=" * 60)
    
    # We'll accumulate judging tasks as we generate
    all_judge_tasks = []
    direct_results = []
    reasoning_results = []
    direct_extracted_all = []
    reasoning_extracted = []
    
    async def judge_batch(batch_tasks):
        """Judge a batch of answers with rate limiting."""
        CHUNK_SIZE = 100
        all_results = []
        for chunk_idx in range(0, len(batch_tasks), CHUNK_SIZE):
            chunk = batch_tasks[chunk_idx:chunk_idx + CHUNK_SIZE]
            print(f"      Judging chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(batch_tasks) + CHUNK_SIZE - 1)//CHUNK_SIZE} ({len(chunk)} requests)...")
            chunk_results = await asyncio.gather(*chunk)
            all_results.extend(chunk_results)
            if chunk_idx + CHUNK_SIZE < len(batch_tasks):
                await asyncio.sleep(1.0)
        return all_results
    
    # ==========================================================================
    # Step 1: Direct answers - generate and judge in batches
    # ==========================================================================
    print(f"\nStep 1: DIRECT answers (no reasoning) - {args.direct_runs} times per question")
    print("-" * 60)
    
    # Generate prompts: each question repeated args.direct_runs times
    direct_prompts = []
    for p in problems:
        msg = DIRECT_PROMPT.format(question=p.question)
        messages = [{"role": "user", "content": msg}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for _ in range(args.direct_runs):
            direct_prompts.append(formatted)
    
    print(f"  Generating {len(direct_prompts)} direct answers...")
    direct_outputs = llm.generate(direct_prompts, direct_sampling)
    
    print("  Extracting and judging direct answers...")
    # Extract and prepare judging tasks immediately
    direct_judge_tasks = []
    for i, p in enumerate(problems):
        attempts = []
        for run_idx in range(args.direct_runs):
            prompt_idx = i * args.direct_runs + run_idx
            text = direct_outputs[prompt_idx].outputs[0].text.strip()
            extracted = extract_direct_answer(text)
            direct_extracted_all.append(extracted)
            attempts.append({
                "run": run_idx + 1,
                "raw_output": text,
            })
            # Create judging task
            direct_judge_tasks.append(
                judge_correctness_async(async_client, args.judge_model, p.answer, extracted)
            )
        direct_results.append({
            "question": p.question,
            "expected": p.answer,
            "attempts": attempts,
        })
    
    # Judge direct answers
    direct_judgments_all = asyncio.run(judge_batch(direct_judge_tasks))
    print(f"  ✓ Direct: {sum(direct_judgments_all)}/{len(direct_judgments_all)} correct")
    
    # ==========================================================================
    # Step 2: Reasoning answers - generate and judge in batches
    # ==========================================================================
    print(f"\nStep 2: REASONING answers (CoT)")
    print("-" * 60)
    
    reasoning_prompts = []
    for p in problems:
        msg = REASONING_PROMPT.format(question=p.question)
        messages = [{"role": "user", "content": msg}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        reasoning_prompts.append(formatted)
    
    print(f"  Generating {len(reasoning_prompts)} reasoning answers...")
    reasoning_outputs = llm.generate(reasoning_prompts, reasoning_sampling)
    
    print("  Extracting and judging reasoning answers...")
    # Extract and prepare judging tasks immediately
    reasoning_judge_tasks = []
    for p, out in zip(problems, reasoning_outputs):
        text = out.outputs[0].text.strip()
        extracted = extract_final_answer(text)
        reasoning_extracted.append(extracted)
        reasoning_results.append({
            "question": p.question,
            "expected": p.answer,
            "raw_output": text,
        })
        # Create judging task
        reasoning_judge_tasks.append(
            judge_correctness_async(async_client, args.judge_model, p.answer, extracted)
        )
    
    # Judge reasoning answers
    reasoning_correct = asyncio.run(judge_batch(reasoning_judge_tasks))
    print(f"  ✓ Reasoning: {sum(reasoning_correct)}/{len(reasoning_correct)} correct")
    
    # ==========================================================================
    # Step 3: Print extraction stats
    # ==========================================================================
    print(f"\nExtraction stats:")
    direct_success = sum(1 for e in direct_extracted_all if e is not None)
    reasoning_success = sum(1 for e in reasoning_extracted if e is not None)
    print(f"  Direct extraction: {direct_success}/{len(direct_extracted_all)} successful")
    print(f"  Reasoning extraction: {reasoning_success}/{len(reasoning_extracted)} successful")
    
    # ==========================================================================
    # Step 4: Group judgments and categorize
    # ==========================================================================
    # Group direct judgments by question - a question is "correct" if ANY attempt succeeds
    direct_correct = []
    direct_attempt_details = []
    for i, r in enumerate(direct_results):
        question_judgments = direct_judgments_all[i * args.direct_runs:(i + 1) * args.direct_runs]
        question_extracted = direct_extracted_all[i * args.direct_runs:(i + 1) * args.direct_runs]
        
        # Store details for each attempt
        attempt_details = []
        for j, (correct, extracted) in enumerate(zip(question_judgments, question_extracted)):
            attempt_details.append({
                "run": j + 1,
                "extracted": extracted,
                "correct": correct,
            })
        direct_attempt_details.append(attempt_details)
        
        # Question is "correct" if ANY attempt succeeded
        any_correct = any(question_judgments)
        direct_correct.append(any_correct)
    print("\n" + "=" * 60)
    print("Filtering and categorizing")
    print("=" * 60)
    
    # Categories:
    # - too_easy: ANY direct attempt succeeded (model can solve without reasoning)
    # - reasoning_needed: ALL direct attempts failed AND reasoning succeeded (requires reasoning)
    # - too_hard: ALL direct attempts failed AND reasoning failed (too hard even with reasoning)
    # - reasoning_hurt: ANY direct attempt succeeded BUT reasoning failed (rare edge case)
    
    too_easy = []
    reasoning_needed = []
    too_hard = []
    reasoning_hurt = []
    
    for i, p in enumerate(problems):
        direct_ok = direct_correct[i]  # True if ANY direct attempt succeeded
        reasoning_ok = reasoning_correct[i]
        
        item = {
            "question": p.question,
            "answer": p.answer,
            "metadata": p.metadata,
            "direct_attempts": direct_attempt_details[i],  # All attempts with details
            "direct_correct": direct_ok,  # True if ANY attempt succeeded
            "direct_success_rate": sum(1 for a in direct_attempt_details[i] if a["correct"]) / args.direct_runs,
            "reasoning_response": reasoning_results[i]["raw_output"],
            "reasoning_extracted": reasoning_extracted[i],
            "reasoning_correct": reasoning_ok,
        }
        
        if direct_ok and reasoning_ok:
            too_easy.append(item)
        elif not direct_ok and reasoning_ok:
            reasoning_needed.append(item)
        elif not direct_ok and not reasoning_ok:
            too_hard.append(item)
        else:  # direct_ok and not reasoning_ok
            reasoning_hurt.append(item)
    
    # Print summary
    print(f"\nResults summary:")
    print(f"  Too easy (ANY direct attempt succeeded):    {len(too_easy):3d} ({100*len(too_easy)/len(problems):.1f}%)")
    print(f"  Reasoning needed (ALL direct failed, reasoning succeeded): {len(reasoning_needed):3d} ({100*len(reasoning_needed)/len(problems):.1f}%)")
    print(f"  Too hard (ALL direct failed, reasoning failed):   {len(too_hard):3d} ({100*len(too_hard)/len(problems):.1f}%)")
    print(f"  Reasoning hurt (ANY direct succeeded, reasoning failed):             {len(reasoning_hurt):3d} ({100*len(reasoning_hurt)/len(problems):.1f}%)")
    print(f"  Total:                                  {len(problems):3d}")
    
    # Additional stats
    total_direct_attempts = len(problems) * args.direct_runs
    successful_direct_attempts = sum(sum(1 for a in details if a["correct"]) for details in direct_attempt_details)
    print(f"\n  Direct mode stats: {successful_direct_attempts}/{total_direct_attempts} attempts succeeded ({100*successful_direct_attempts/total_direct_attempts:.1f}%)")
    print(f"  Reasoning mode stats: {sum(reasoning_correct)}/{len(reasoning_correct)} succeeded ({100*sum(reasoning_correct)/len(reasoning_correct):.1f}%)")

    # ==========================================================================
    # Save results
    # ==========================================================================
    output_data = {
        "model": args.model,
        "judge_model": args.judge_model,
        "split": args.split,
        "total_questions": len(problems),
        "summary": {
            "too_easy": len(too_easy),
            "reasoning_needed": len(reasoning_needed),
            "too_hard": len(too_hard),
            "reasoning_hurt": len(reasoning_hurt),
        },
        # The main output: questions that require reasoning
        "reasoning_needed": reasoning_needed,
        # Also save others for reference
        "too_easy": too_easy,
        "too_hard": too_hard,
        "reasoning_hurt": reasoning_hurt,
    }
    
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Use 'reasoning_needed' list ({len(reasoning_needed)} questions) for rat2 experiments")

    # Also save a simple list of just the questions for easy loading
    simple_output = output_path.with_suffix(".questions.json")
    simple_data = [
        {"question": item["question"], "answer": item["answer"]}
        for item in reasoning_needed
    ]
    with open(simple_output, "w") as f:
        json.dump(simple_data, f, indent=2)
    print(f"Simple question list saved to: {simple_output}")


if __name__ == "__main__":
    main()
