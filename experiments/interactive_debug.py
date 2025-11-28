#!/usr/bin/env python3
"""Interactive debugging script for testing model prompts and outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.prompts import EVASION_ACTOR_PROMPT


def extract_think(text: str) -> str | None:
    """Extract thinking content (everything before </think> tag)."""
    if '</think>' in text.lower():
        idx = text.lower().find('</think>')
        return text[:idx].strip() or None
    return None


def extract_answer(text: str) -> str | None:
    """Extract content between <answer> and </answer> tags (last instance)."""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def print_colored(text: str, color: str) -> None:
    """Print text with ANSI color."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "dim": "\033[90m",  # Gray/dim for prefill
        "reset": "\033[0m",
        "bold": "\033[1m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def main():
    parser = argparse.ArgumentParser(description="Interactive model debugging")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", help="Model ID")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model context length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size")
    args = parser.parse_args()

    print_colored(f"\n{'='*60}", "cyan")
    print_colored("Interactive Model Debugger", "bold")
    print_colored(f"{'='*60}\n", "cyan")
    
    print(f"Loading model: {args.model}")
    print("This may take a few minutes...\n")
    
    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
    }
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.9,
        stop=["</answer>"],  # Stop after answer tag
    )
    
    print_colored("Model loaded successfully!\n", "green")
    print_colored("Commands:", "yellow")
    print("  /system           - Enter multi-line system prompt (end with empty line)")
    print("  /prefill          - Enter multi-line prefill for model output (end with empty line)")
    print("  /temp <value>     - Set temperature (e.g., /temp 0.3)")
    print("  /tokens <value>   - Set max tokens (e.g., /tokens 4096)")
    print("  /show             - Show current settings")
    print("  /clear            - Clear system prompt and prefill")
    print("  /raw              - Toggle showing raw output")
    print("  /quit             - Exit")
    print("  <anything else>   - Send as user message")
    print()
    
    # Default to evasion prompt (without the "Question:\n{question}" part)
    default_system = EVASION_ACTOR_PROMPT.replace("\n\nQuestion:\n{question}", "").strip()
    system_prompt = default_system
    prefill = ""
    show_raw = True  # Always show raw by default for debugging
    
    print_colored("Loaded default system prompt (EVASION_ACTOR_PROMPT)", "green")
    print_colored(f"Preview: {system_prompt[:150]}...\n", "dim")
    
    while True:
        try:
            print_colored("─" * 60, "cyan")
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break
                    
                elif cmd == "/system":
                    print_colored("Enter system prompt (empty line to finish):", "yellow")
                    lines = []
                    while True:
                        line = input("")
                        if line == "":
                            break
                        lines.append(line)
                    system_prompt = "\n".join(lines)
                    print_colored(f"System prompt set to:\n{system_prompt}", "green")
                    continue
                
                elif cmd == "/prefill":
                    print_colored("Enter prefill text (empty line to finish):", "yellow")
                    lines = []
                    while True:
                        line = input("")
                        if line == "":
                            break
                        lines.append(line)
                    prefill = "\n".join(lines)
                    print_colored(f"Prefill set to:\n{prefill}", "dim")
                    continue
                    
                elif cmd == "/clear":
                    system_prompt = ""
                    prefill = ""
                    print_colored("System prompt and prefill cleared.", "green")
                    continue
                    
                elif cmd == "/temp":
                    try:
                        sampling_params.temperature = float(arg)
                        print_colored(f"Temperature set to {sampling_params.temperature}", "green")
                    except ValueError:
                        print_colored("Invalid temperature value", "red")
                    continue
                    
                elif cmd == "/tokens":
                    try:
                        sampling_params.max_tokens = int(arg)
                        print_colored(f"Max tokens set to {sampling_params.max_tokens}", "green")
                    except ValueError:
                        print_colored("Invalid token count", "red")
                    continue
                    
                elif cmd == "/show":
                    print_colored("\nCurrent Settings:", "yellow")
                    print(f"  Model: {args.model}")
                    print(f"  Temperature: {sampling_params.temperature}")
                    print(f"  Max tokens: {sampling_params.max_tokens}")
                    print(f"  Show raw: {show_raw}")
                    print(f"  System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"  System prompt: {system_prompt or '(none)'}")
                    print(f"  Prefill: {prefill[:100]}..." if len(prefill) > 100 else f"  Prefill: {prefill or '(none)'}")
                    continue
                
                elif cmd == "/raw":
                    show_raw = not show_raw
                    print_colored(f"Show raw output: {show_raw}", "green")
                    continue
                    
                else:
                    print_colored(f"Unknown command: {cmd}", "red")
                    continue
            
            # Build messages for chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_input})
            
            # Apply chat template
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            print_colored("\n[Formatted Prompt Preview]", "magenta")
            # Show first 500 chars of formatted prompt
            preview = full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt
            print(preview)
            
            print_colored("\n[Generating...]", "yellow")
            
            # Add prefill to prompt if set
            if prefill:
                full_prompt += prefill
            
            # Generate
            outputs = llm.generate([full_prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            token_count = len(outputs[0].outputs[0].token_ids)
            
            # Combine prefill + response for parsing
            full_output = prefill + response_text if prefill else response_text
            
            # Parse from full output (prefill + response)
            think_content = extract_think(full_output)
            answer_content = extract_answer(full_output)
            
            print_colored(f"\n{'─'*60}", "cyan")
            print_colored(f"TOKENS: {token_count}", "magenta")
            print_colored(f"{'─'*60}", "cyan")
            
            # Always show raw output first if enabled
            if show_raw:
                print_colored("\n[RAW OUTPUT]", "yellow")
                if prefill:
                    print_colored(prefill, "dim")  # Show prefill in dim/gray
                print(response_text)
                print_colored(f"\n{'─'*60}", "cyan")
            
            # Then show parsed sections
            if think_content:
                print_colored("\n[PARSED <think>]", "blue")
                # Show prefill portion in dim if it's part of think content
                if prefill and think_content.startswith(prefill.strip()):
                    print_colored(prefill.strip(), "dim")
                    print(think_content[len(prefill.strip()):].strip())
                else:
                    print(think_content)
            
            if answer_content:
                print_colored("\n[PARSED <answer>]", "green")
                print(answer_content)
            
            if not think_content and not answer_content and not show_raw:
                print_colored("\n[NO TAGS FOUND - RAW OUTPUT]", "yellow")
                if prefill:
                    print_colored(prefill, "dim")
                print(response_text)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
            continue
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

