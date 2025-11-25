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

