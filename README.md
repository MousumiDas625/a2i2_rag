# A2I2 Final — IQL-based Operator Policy Selection for Wildfire Evacuation Dialogues

This project trains an **Implicit Q-Learning (IQL)** policy selector that learns to pick the best persuasion strategy for an emergency operator trying to convince different residents to evacuate during a wildfire. It includes three experiment baselines, a conversation simulation engine, and a FastAPI server.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Pipeline Overview](#pipeline-overview)
4. [How to Run](#how-to-run)
5. [Experiments](#experiments)
6. [Where to Find Outputs & Evaluation](#where-to-find-outputs--evaluation)
7. [Switching LLM Providers](#switching-llm-providers)
8. [Interactive Mode](#interactive-mode)
9. [API Server](#api-server)

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ (tested with 3.12) |
| **Conda env** | `a2i2` — already has all dependencies |
| **Ollama** | Running locally with `llama3` pulled |
| **GPU** | CUDA or Apple MPS (auto-detected; falls back to CPU) |

Activate the environment:

```bash
conda activate a2i2
```

Make sure Ollama is running (needed only for experiments/simulation):

```bash
ollama serve          # in a separate terminal
ollama pull llama3    # one-time download
```

---

## Project Structure

```
a2i2_final/
├── config/
│   ├── settings.py          # All paths, hyperparameters, device, LLM config
│   └── personas.py          # Resident persona definitions (10 personas)
│
├── preprocessing/            # STAGE 1: Data preparation
│   ├── P01_xlsx_to_jsonl.py  # Convert raw XLSX transcripts → JSONL
│   ├── P02_clean_and_merge.py# Drop Julie, merge consecutive turns, normalize
│   ├── P03_extract_residents.py # Discover unique resident names
│   └── P04_add_rewards.py    # Add reward=1 to last resident turn per dialogue
│
├── iql/                      # STAGE 2: IQL training pipeline
│   ├── networks.py           # Q-network & V-network architectures (PyTorch)
│   ├── I01_build_iql_dataset.py   # Build (state, action, reward) tuples
│   ├── I02_build_operator_policies.py # Compute per-resident policy prototypes
│   ├── I03_train_iql.py      # Train IQL Q-network and V-network
│   ├── I04_build_rag_indexes.py   # Build per-policy FAISS indexes
│   └── I05_extract_successful_utterances.py # Build global successful-ops corpus
│
├── retrieval/                # Runtime inference modules
│   ├── policy_selector.py    # Load trained IQL model, select best policy
│   └── rag_retrieval.py      # Retrieve few-shot examples from FAISS
│
├── simulation/               # Conversation engine
│   ├── llm_client.py         # Unified LLM client (Ollama / OpenAI)
│   ├── operator_generator.py # Generate operator utterances (3 strategies)
│   ├── resident_simulator.py # Generate resident utterances via persona LLM
│   ├── decision.py           # Judge conversation success/failure
│   ├── conversation_loop.py  # Core turn-by-turn simulation loop
│   └── interactive.py        # Interactive console mode
│
├── experiments/              # STAGE 3 & 4: Run and evaluate experiments
│   ├── exp1_zero_shot.py     # Experiment 1: Zero-shot baseline
│   ├── exp2_rag_successful.py # Experiment 2: RAG over successful operators
│   ├── exp3_iql_policy.py    # Experiment 3: IQL policy selection + RAG
│   ├── batch_runner.py       # Run all experiments in batch
│   └── evaluate.py           # Compare results across experiments
│
├── api/
│   └── server.py             # FastAPI server with /chat, /simulate endpoints
│
├── data/                     # All generated data (created by the pipeline)
│   ├── raw_xlsx/             # Source: 104 XLSX transcription files
│   ├── jsonl/                # Stage 1 output: per-dialogue JSONL
│   ├── cleaned/              # Stage 1 output: cleaned/merged dialogues
│   ├── meta/                 # residents.json (unique resident list)
│   ├── reports/              # Processing reports
│   ├── iql/                  # Stage 2 output: dataset + trained models
│   │   ├── iql_dataset.jsonl
│   │   ├── label_map.json
│   │   ├── config.json
│   │   └── selector/         # Trained model weights + plots
│   ├── indexes/              # FAISS indexes + policy prototypes
│   │   ├── policies/
│   │   └── faiss/
│   ├── successful_ops/       # Successful operator utterance corpus
│   └── runs/                 # Experiment results (per-run logs + summaries)
│
├── run_pipeline.sh           # One-command full pipeline script
└── requirements.txt          # Python dependencies
```

---

## Pipeline Overview

The pipeline has **4 stages**. Stages 1 and 2 are offline (data processing + training). Stages 3 and 4 require Ollama running.

### Stage 1: Preprocessing (P01–P04)

| Script | What it does | Output |
|---|---|---|
| `P01_xlsx_to_jsonl.py` | Reads 104 raw XLSX transcripts, converts each to JSONL | `data/jsonl/` |
| `P02_clean_and_merge.py` | Drops Julie turns, normalizes text, merges consecutive same-role turns | `data/cleaned/` |
| `P03_extract_residents.py` | Finds unique resident names across all dialogues | `data/meta/residents.json` |
| `P04_add_rewards.py` | Assigns reward=1 to the last resident turn per dialogue (evacuation success signal) | Updates `data/cleaned/` in-place |

### Stage 2: IQL Pipeline (I01–I05)

| Script | What it does | Output |
|---|---|---|
| `I01_build_iql_dataset.py` | Encodes states via sentence embeddings, builds (state, action, reward) tuples | `data/iql/iql_dataset.jsonl` |
| `I02_build_operator_policies.py` | Computes per-resident operator policy prototypes (centroid embeddings) | `data/indexes/policies/` |
| `I03_train_iql.py` | Trains Q-network and V-network with Bellman updates + early stopping | `data/iql/selector/*.pt` |
| `I04_build_rag_indexes.py` | Builds per-policy FAISS indexes for few-shot RAG retrieval | `data/indexes/faiss/` |
| `I05_extract_successful_utterances.py` | Builds a global corpus of operator utterances from successful dialogues | `data/successful_ops/` |

### Stage 3: Experiments (exp1–exp3)

| Experiment | Strategy | Description |
|---|---|---|
| **Exp 1: Zero-Shot** | `zero_shot` | LLM generates operator responses with no training data — pure instruction following |
| **Exp 2: RAG-Successful** | `rag_successful` | Retrieves similar utterances from ALL successful operators as few-shot examples |
| **Exp 3: IQL + RAG** | `iql_rag` | IQL selects the best policy, then RAG retrieves from that policy's index |

### Stage 4: Evaluation

`evaluate.py` reads the `summary.json` from each experiment and prints a comparison table + writes `data/runs/comparison.csv`.

---

## How to Run

### Option A: Run the full pipeline (one command)

```bash
conda activate a2i2
cd /Users/mousumi/biyik_1165/a2i2_new/A2I2_Chatbot/a2i2_final

# Make sure Ollama is serving llama3 in another terminal
bash run_pipeline.sh
```

### Option B: Run step by step

```bash
conda activate a2i2
cd /Users/mousumi/biyik_1165/a2i2_new/A2I2_Chatbot/a2i2_final

# ── Stage 1: Preprocessing ──
python preprocessing/P01_xlsx_to_jsonl.py
python preprocessing/P02_clean_and_merge.py
python preprocessing/P03_extract_residents.py
python preprocessing/P04_add_rewards.py

# ── Stage 2: IQL Training ──
python iql/I01_build_iql_dataset.py
python iql/I02_build_operator_policies.py
python iql/I03_train_iql.py --mode state    # or --mode embed
python iql/I04_build_rag_indexes.py
python iql/I05_extract_successful_utterances.py

# ── Stage 3: Run Experiments (requires Ollama) ──
python experiments/exp1_zero_shot.py --runs 5
python experiments/exp2_rag_successful.py --runs 5
python experiments/exp3_iql_policy.py --runs 5 --mode state

# ── Stage 4: Evaluation ──
python experiments/evaluate.py
```

### Experiment options

All experiment scripts accept:

| Flag | Default | Description |
|---|---|---|
| `--residents` | all | Comma-separated resident names (e.g. `bob,ross`) |
| `--runs` | 5 | Number of conversations per resident |
| `--max-turns` | 16 | Max turns before forced stop |
| `--seed` | "Hello, this is the fire department..." | Opening operator line |
| `--mode` (exp3 only) | state | IQL Q-network mode: `state` or `embed` |

---

## Where to Find Outputs & Evaluation

### Training Artifacts

| Artifact | Path |
|---|---|
| Trained Q-network (state mode) | `data/iql/selector/iql_model_state.pt` |
| Trained Q-network (embed mode) | `data/iql/selector/iql_model_embed.pt` |
| Trained V-network (state mode) | `data/iql/selector/value_model_state.pt` |
| Trained V-network (embed mode) | `data/iql/selector/value_model_embed.pt` |
| Training loss curves (state) | `data/iql/selector/training_curves_state.png` |
| Training loss curves (embed) | `data/iql/selector/training_curves_embed.png` |
| Per-policy Q-values (state) | `data/iql/selector/per_policy_qvalues_state.png` |
| Per-policy Q-values (embed) | `data/iql/selector/per_policy_qvalues_embed.png` |
| IQL dataset | `data/iql/iql_dataset.jsonl` (383 samples) |
| Label map | `data/iql/label_map.json` |

### Experiment Results

Each experiment run creates a timestamped folder under `data/runs/`:

```
data/runs/
├── exp1_zero_shot_20260408T.../
│   └── summary.json              # Per-resident success rates + individual run details
├── exp2_rag_successful_20260408T.../
│   └── summary.json
├── exp3_iql_policy_20260408T.../
│   └── summary.json
└── comparison.csv                # Side-by-side comparison of all 3 experiments
```

Each `summary.json` contains:
- `per_resident`: success counts and rates per resident
- `overall_success_rate`: aggregate success rate
- `results[]`: individual run details (status, turns, transcript path)

### Evaluation

Run `python experiments/evaluate.py` after all three experiments to get a printed comparison table and `data/runs/comparison.csv`.

---

## Switching LLM Providers

The default is **Ollama with llama3**. To switch to OpenAI GPT (or any compatible API):

### Via environment variables (recommended):

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o
# Then run experiments normally
```

### Via config/settings.py:

Edit `config/settings.py` and change:

```python
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o"
```

The `simulation/llm_client.py` module dispatches calls to the right backend automatically.

---

## Interactive Mode

For human-in-the-loop conversations:

```bash
python simulation/interactive.py
```

This starts a console session where you can play as either the operator or the resident while the LLM handles the other role.

---

## API Server

Start the FastAPI server:

```bash
python api/server.py
# or: uvicorn api.server:app --host 0.0.0.0 --port 8001
```

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/personas` | GET | List available resident personas |
| `/chat` | POST | Single-turn resident response given conversation history |
| `/simulate` | POST | Run a full multi-turn simulation |

---

## Training Summary (Current Run)

- **Dataset**: 104 dialogues, 383 IQL training samples, 5 residents (bob, lindsay, michelle, niki, ross)
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim)
- **Device**: MPS (Apple Silicon GPU)
- **State-mode training**: Early stopped at epoch 40 (best val loss: 0.684 at epoch 20)
- **Embed-mode training**: Ran full 200 epochs (best val loss: 0.466 at epoch 199)
- **Successful dialogues**: 98 out of 104 (94.2%)
- **Successful operator utterances corpus**: 312 utterances with FAISS index
