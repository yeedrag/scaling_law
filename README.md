## CoT Monitorability Scaling Experiment

This repository replicates the chemistry setting from *When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors* and measures how reasoning length correlates with a model's ability to evade a CoT monitor.

### Requirements

- Python 3.10+
- NVIDIA GPU (tested with A100 80GB)
- Access to the Qwen3-32B thinking model via vLLM
- OpenAI API key for the monitor model

Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.yaml` to adjust:

- Hugging Face dataset (`rl-actors/Chemistry-GPQA` by default)
- Temperature (default `0.6`)
- Samples per problem
- Qwen model id and vLLM settings
- OpenAI monitor model (default `gpt-4o-mini`)

Set required environment variables:

```bash
export OPENAI_API_KEY=sk-...
```

### Running the experiment

```bash
python experiments/run_experiment.py --config config.yaml --tensor-parallel-size 2
```

Optional flags:

- `--limit N` – run on first `N` problems for smoke testing.
- `--log-level DEBUG` – verbose logging.

Outputs:

- `results/chemistry_monitorability.jsonl` – per-sample records with reasoning tokens, accuracy, monitor decisions.
- `results/summary.json` – aggregate metrics (baseline accuracy, evasion success, monitor rate, average reasoning length).

### Next steps

Use the generated JSONL to analyze the correlation between baseline reasoning length and evasion success. A starter notebook can be added under `experiments/analysis.ipynb`.

## RAT2: Monitorability-focused red teaming

`experiments/rat2.py` implements a second automatic red-teaming loop that targets the **legibility** and **coverage** metrics defined in *A Pragmatic Way to Measure Chain-of-Thought Monitorability* (arXiv:2510.23966). The script:

- Samples GSM8K problems via `load_problems`.
- Runs a target reasoning model with an adversarial prompt that encourages hard-to-monitor chain-of-thought traces.
- Scores only the correct answers with the autorater prompt from the paper (implemented in `MONITORABILITY_AUTORATER_PROMPT`).
- Tracks accuracy, average legibility/coverage, and reasoning length (token/character counts).
- Uses an optimizer model to rewrite the prompt to further reduce monitorability while keeping answers correct.

Example:

```bash
python experiments/rat2.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --optimizer gpt-4o \
  --autorater-model gpt-4o-mini \
  --dataset-limit 8 \
  --iterations 5
```

Results are saved to `rat2_results.json`, including per-iteration metrics, per-sample autorater scores, and the best-performing prompt.

### RAT2: Monitorability-focused prompt search

`experiments/rat2.py` runs an automatic red-teaming loop that tries to **minimize the legibility and coverage metrics** from *A Pragmatic Way to Measure Chain-of-Thought Monitorability* while keeping GSM8K answers correct.

Highlights:

- Loads GSM8K problems via `datasets`.
- Runs the target reasoning model with vLLM, logs `<think>` reasoning, answers, and reasoning length (tokens + characters).
- Scores each correct reasoning trace with the autorater prompt from the paper, capturing legibility (0–4) and coverage (0–4).
- Uses an OpenAI optimizer model to iteratively rewrite the red-team prompt to drive monitorability lower without hurting accuracy.
- Saves a full trace (per-sample metrics plus the best prompt) to a JSON file, default `rat2_results.json`.

Example command:

```bash
python experiments/rat2.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --optimizer gpt-4o \
  --autorater-model gpt-4o-mini \
  --iterations 5 \
  --dataset-limit 8
```

You will need `OPENAI_API_KEY` set for both the optimizer and autorater calls.

