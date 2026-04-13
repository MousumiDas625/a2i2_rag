# A2I2 Final — IQL-based Operator Policy Selection for Wildfire Evacuation Dialogues

This project trains an **Implicit Q-Learning (IQL)** policy selector that learns to pick the best persuasion strategy for an emergency operator trying to convince different residents to evacuate during a wildfire. It includes three experiment baselines, a conversation simulation engine, and a FastAPI server.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup (uv)](#environment-setup-uv)
3. [OpenAI API Key](#openai-api-key)
4. [Project Structure](#project-structure)
5. [Resident Personas](#resident-personas)
6. [Pipeline Overview](#pipeline-overview)
7. [How to Run](#how-to-run)
8. [Running on Endeavour (SLURM)](#running-on-endeavour-slurm)
9. [Success Rate Matrices](#success-rate-matrices)
10. [Where to Find Outputs](#where-to-find-outputs)
11. [IQL Dueling-Embed Architecture](#iql-dueling-embed-architecture)
12. [Conversation Engine](#conversation-engine)
13. [Decision Judge](#decision-judge)
14. [Switching LLM Providers](#switching-llm-providers)
15. [Interactive Mode](#interactive-mode)
16. [API Server](#api-server)

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10+ |
| **uv** | Fast Python package manager — installed by `slurm/setup_env.sh` |
| **OpenAI API key** | For LLM calls during experiments (see below) |
| **GPU** | CUDA (cluster) or Apple MPS (local) — auto-detected; falls back to CPU |
| **curl** | For installing uv (pre-installed on all Linux systems) |

> **Ollama alternative**: If you prefer running a local LLM instead of OpenAI, install Ollama, pull `llama3`, run `ollama serve`, and set `LLM_PROVIDER=ollama` (see [Switching LLM Providers](#switching-llm-providers)).

---

## Environment Setup (uv)

[uv](https://github.com/astral-sh/uv) is a fast, self-contained Python package manager — no conda or system Python required.

### Local machine (macOS / Linux)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.zshrc or ~/.bashrc permanently

# 2. Create the virtual environment
cd /path/to/a2i2_final
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install dependencies
uv pip install torch                     # CPU torch is fine for local dev
uv pip install -r requirements.txt

# 4. Activate whenever you work on the project
source .venv/bin/activate
```

### Endeavour HPC cluster

Run the one-time setup script **on a login node** before submitting any SLURM jobs:

```bash
cd /project2/biyik_1165/mousumid/a2i2_new/A2I2_Chatbot/a2i2_final
bash slurm/setup_env.sh
```

This script automatically:
- Downloads and installs `uv` to `~/.local/bin/`
- Creates `.venv/` with Python 3.10
- Installs PyTorch with CUDA 12.1 support
- Installs all dependencies and replaces `faiss-cpu` with `faiss-gpu`
- Verifies CUDA is visible to PyTorch

You only need to run this **once**. All SLURM job scripts then activate `.venv` automatically.

---

## OpenAI API Key (optional — only if not using Ollama)

If you prefer OpenAI over Ollama, follow these steps:

1. Go to [platform.openai.com](https://platform.openai.com) → sign in → **API keys** → **Create new secret key**. Copy the key (`sk-proj-...`).
2. Add billing at **Billing** → add a payment method. Estimated cost for all experiments: **$5–10** using `gpt-4o-mini`.
3. Set it on the cluster:

```bash
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
source ~/.bashrc
```

Then in each `slurm/run_exp*.sh`, comment out the Ollama block and uncomment the OpenAI block (see [Running on Endeavour](#running-on-endeavour-slurm)).

---

## Project Structure

```
a2i2_final/
├── config/
│   ├── settings.py          # All paths, hyperparameters, device, LLM config
│   └── personas.py          # 10 resident persona definitions
│
├── preprocessing/            # STAGE 1: Data preparation
│   ├── P01_xlsx_to_jsonl.py  # XLSX transcripts → per-dialogue JSONL
│   ├── P02_clean_and_merge.py # Drop Julie, map roles, merge consecutive turns
│   ├── P03_extract_residents.py # Unique residents → meta/residents.json
│   └── P04_add_rewards.py    # Add reward=1 to last resident turn per dialogue
│
├── iql/                      # STAGE 2: IQL training pipeline
│   ├── networks.py           # QNetworkEmbed, QNetworkState, ValueNetwork
│   ├── I01_build_iql_dataset.py
│   ├── I02_build_operator_policies.py
│   ├── I03_train_iql.py
│   ├── I04_build_rag_indexes.py
│   ├── I05_extract_successful_utterances.py
│   └── plots/                # Training curves and per-policy Q-value plots
│
├── retrieval/
│   ├── policy_selector.py    # Load trained IQL model, select best policy
│   └── rag_retrieval.py      # Retrieve few-shot examples from FAISS indexes
│
├── simulation/               # Conversation engine
│   ├── llm_client.py         # Unified LLM client (Ollama / OpenAI)
│   ├── operator_generator.py # Generate operator utterances (3 strategies)
│   ├── resident_simulator.py # Generate resident utterances via persona LLM
│   ├── decision.py           # Judge conversation success/failure
│   ├── conversation_loop.py  # Core turn-by-turn simulation loop
│   └── interactive.py        # Interactive console mode
│
├── experiments/              # STAGE 3 & 4: Run and evaluate
│   ├── exp1_zero_shot.py
│   ├── exp2_rag_successful.py
│   ├── exp3_iql_policy.py
│   ├── batch_runner.py       # Grid over seeds × residents × repetitions
│   ├── evaluate.py           # Side-by-side comparison table + CSV
│   └── make_success_matrices.py  # Per-experiment success matrices + heatmaps
│
├── api/
│   └── server.py             # FastAPI HTTP server (health, personas, chat, simulate)
│
├── slurm/                    # Endeavour HPC job scripts
│   ├── setup_env.sh          # One-time environment setup (run on login node)
│   ├── lib_ollama.sh         # Ollama start/stop helpers for SLURM jobs
│   ├── ollama_env.sh         # Shared Ollama config (generated by setup_env.sh)
│   ├── submit_all.sh         # Submit all 3 experiments + matrix job
│   ├── run_exp1_zero_shot.sh
│   ├── run_exp2_rag.sh
│   └── run_exp3_iql.sh
│
├── data/                     # All generated data (created by the pipeline)
│   ├── raw_xlsx/             # Source XLSX transcripts
│   ├── jsonl/                # Per-dialogue JSONL + combined.jsonl
│   ├── cleaned/              # Cleaned dialogues with rewards
│   ├── meta/                 # residents.json
│   ├── reports/              # Preprocessing summaries
│   ├── iql/                  # IQL dataset, config, label map
│   │   └── selector/         # Trained model weights (.pt) + norm_stats.npz
│   ├── indexes/
│   │   ├── policies/         # Per-policy operator prototypes, pairs, embeddings
│   │   └── faiss/            # Per-policy FAISS indexes for RAG retrieval
│   ├── successful_ops/       # Global successful-operator corpus + FAISS index
│   └── runs/                 # Experiment results, summaries, success matrices
│
├── .venv/                    # Virtual environment (created by setup_env.sh)
├── run_pipeline.sh           # Full end-to-end pipeline script
└── requirements.txt
```

---

## Resident Personas

| Persona | Has Training Data | Description |
|---|---|---|
| bob | **Yes** | Stubborn, prioritizes work over safety |
| niki | **Yes** | Cooperative, willing to follow instructions |
| lindsay | **Yes** | Babysitter responsible for children |
| michelle | **Yes** | Stubborn, determined to protect property |
| ross | **Yes** | Van driver helping evacuate elderly passengers |
| mary | No | Elderly person living alone with a pet |
| ben | No | Young professional working from home |
| ana | No | Caregiver at a senior center |
| tom | No | Community-connected, wants to help others first |
| mia | No | Teen student absorbed in a school robotics project |

The 5 "core" personas have human-collected dialogues used for IQL training. The 5 "extended" personas are LLM-simulated only and test generalization to unseen personality types.

Each persona is defined in `config/personas.py` with a `scenario` (second-person narrative) and `information` (bullet-point facts) that ground the LLM's behaviour during resident simulation.

---

## Pipeline Overview

### Stage 1: Preprocessing (P01–P04)

| Script | What it does | Output |
|---|---|---|
| `P01_xlsx_to_jsonl.py` | Converts XLSX transcripts to per-dialogue JSONL + `combined.jsonl` | `data/jsonl/` |
| `P02_clean_and_merge.py` | Drops Julie/system, maps speakers to roles, merges consecutive same-role turns | `data/cleaned/` |
| `P03_extract_residents.py` | Extracts unique resident names | `data/meta/residents.json` |
| `P04_add_rewards.py` | Adds `reward=1` to last resident turn per dialogue, 0 elsewhere | Updates `data/cleaned/` in-place |

### Stage 2: IQL Pipeline (I01–I05)

| Script | What it does | Output |
|---|---|---|
| `I01_build_iql_dataset.py` | Encodes states (mean embedding of last N resident utterances), builds `(state, action, reward, next_state)` tuples | `data/iql/iql_dataset.jsonl`, `config.json`, `label_map.json` |
| `I02_build_operator_policies.py` | Computes per-policy operator prototype embeddings (centroid of utterances per resident) and resident→operator pair files | `data/indexes/policies/` |
| `I03_train_iql.py` | Trains Dueling-Embed Q-network + V-network with EMA target, margin loss, and orthogonality regularization | `data/iql/selector/*.pt`, `norm_stats.npz`, `iql/plots/` |
| `I04_build_rag_indexes.py` | Builds per-policy FAISS `IndexFlatIP` indexes over L2-normalized resident-text embeddings for few-shot RAG retrieval | `data/indexes/faiss/` |
| `I05_extract_successful_utterances.py` | Extracts all operator utterances from successful dialogues into a global corpus + FAISS index | `data/successful_ops/` |

### Stage 3: Experiments

| Experiment | Strategy | Description | Dependencies |
|---|---|---|---|
| **Exp 1** | `zero_shot` | No training data — pure instruction following | LLM + personas only |
| **Exp 2** | `rag_successful` | Retrieves from global successful-operator corpus | I05 |
| **Exp 3** | `iql_rag` | IQL selects best policy → RAG from that policy's index | I01–I04 + trained weights |

### Stage 4: Evaluation

- `evaluate.py` — Prints a comparison table across all three experiments and writes `data/runs/comparison.csv`.
- `make_success_matrices.py` — Generates per-experiment success matrices (CSV + PNG heatmaps).

---

## How to Run

### Option A: Full pipeline (one command)

```bash
source .venv/bin/activate
cd /path/to/a2i2_final

export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

bash run_pipeline.sh
```

### Option B: Step by step

```bash
source .venv/bin/activate
cd /path/to/a2i2_final

export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."

# Stage 1: Preprocessing
python preprocessing/P01_xlsx_to_jsonl.py
python preprocessing/P02_clean_and_merge.py
python preprocessing/P03_extract_residents.py
python preprocessing/P04_add_rewards.py

# Stage 2: IQL Pipeline
python iql/I01_build_iql_dataset.py
python iql/I02_build_operator_policies.py
python iql/I03_train_iql.py
python iql/I04_build_rag_indexes.py
python iql/I05_extract_successful_utterances.py

# Stage 3: Experiments (3 runs each by default)
python experiments/exp1_zero_shot.py --runs 3
python experiments/exp2_rag_successful.py --runs 3
python experiments/exp3_iql_policy.py --runs 3

# Stage 4: Evaluation
python experiments/evaluate.py
python experiments/make_success_matrices.py
```

### Experiment flags

| Flag | Default | Description |
|---|---|---|
| `--residents` | all 10 | Comma-separated names, e.g. `bob,ross` |
| `--runs` | 5 | Conversations per resident |
| `--max-turns` | 16 | Hard turn limit |
| `--seed` | "Hello, this is the fire department…" | Opening operator line |
| `--tag` | (none) | Suffix for output directory name (exp3 only) |

---

## Running on Endeavour (SLURM)

### Step 1 — One-time setup (login node)

Run this **once** before submitting any jobs. It installs `uv`, creates the venv, installs all Python dependencies, downloads the Ollama binary, and pulls the llama3 model (~4.7 GB) into project space:

```bash
cd /project2/biyik_1165/mousumid/a2i2_new/A2I2_Chatbot/a2i2_final
bash slurm/setup_env.sh
```

What happens under the hood:
- `uv` → `~/.local/bin/uv`
- Python 3.10 venv → `a2i2_final/.venv/`
- Ollama binary → `a2i2_final/.ollama/ollama`
- llama3 model weights → `/project2/biyik_1165/mousumid/a2i2_new/ollama_models/` (project space, not home directory quota)
- Shared config → `slurm/ollama_env.sh` (read by job scripts automatically)

### Step 2 — Fill in your account in each SLURM script

Open each `slurm/run_exp*.sh` and replace `<YOUR_ACCOUNT>` and `<YOUR_EMAIL>`:

```bash
#SBATCH --account=ttrojan_123       # find yours with: myquota
#SBATCH --mail-user=you@usc.edu
```

### Step 3 — Submit everything

```bash
bash slurm/submit_all.sh
```

This submits all three experiments **in parallel**, then automatically queues `make_success_matrices.py` with `--dependency=afterok` so it only runs once all three finish.

### How Ollama runs inside each job

Each SLURM script sources `slurm/lib_ollama.sh` and calls `ollama_start`, which:

1. Picks a free port (avoids collisions if two jobs land on the same node)
2. Starts `ollama serve` as a background process — uses the compute node's GPU automatically via `CUDA_VISIBLE_DEVICES`
3. Waits up to 90 seconds for the server to be ready
4. Exports `OLLAMA_URL` so `llm_client.py` finds it
5. Registers a `trap` so `ollama stop` is called on job exit (clean or error)

Ollama server logs are written to `slurm/logs/ollama_<jobid>.log`.

### Monitor

```bash
squeue -u $USER                              # see running/queued jobs
tail -f slurm/logs/exp1_<jobid>.out         # stream experiment output
tail -f slurm/logs/ollama_<jobid>.log       # stream Ollama server log
```

### GPU usage

Each job requests `--gres=gpu:1`. The GPU is used by:

| Component | Usage |
|---|---|
| **Ollama / llama3** | LLM inference for operator + resident utterances (primary GPU load) |
| **Sentence-transformer** | Embedding resident utterances for RAG + IQL state encoding |
| **IQL Q-network** (Exp 3) | Policy selection inference |
| **FAISS-GPU** | Nearest-neighbour retrieval from policy indexes |

### Switching to OpenAI instead of Ollama

If you prefer the OpenAI API (no GPU needed for LLM, lower latency), comment out the Ollama block in each job script and uncomment the OpenAI block:

```bash
# In slurm/run_exp1_zero_shot.sh (and exp2, exp3):

# Comment out:
# source "$(dirname "${BASH_SOURCE[0]}")/lib_ollama.sh"
# ollama_start

# Uncomment:
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."     # or set in ~/.bashrc
export OPENAI_MODEL="gpt-4o-mini"
```

---

## Success Rate Matrices

After all experiments complete, generate per-experiment matrices:

```bash
python experiments/make_success_matrices.py
```

Output in `data/runs/matrices_<timestamp>/`:

| File | Contents |
|---|---|
| `zero_shot_matrix.csv` | Rows = residents, Columns = Run1…RunN, cells = 0/1 |
| `rag_successful_matrix.csv` | Same format for Exp 2 |
| `iql_policy_matrix.csv` | Same format for Exp 3 |
| `zero_shot_matrix.png` | Colour heatmap (green=success, red=fail) |
| `rag_successful_matrix.png` | Same for Exp 2 |
| `iql_policy_matrix.png` | Same for Exp 3 |

Console output example:

```
══════════════════════════════════════════════════════════════
  SUCCESS MATRIX — ZERO-SHOT
══════════════════════════════════════════════════════════════
  Resident        Run 1   Run 2   Run 3   Rate
  ──────────────────────────────────────────────────────────
  ana                 ✓       ✗       ✓    67%
  bob                 ✗       ✗       ✗     0%
  ...
```

You can also pass explicit summary paths:

```bash
python experiments/make_success_matrices.py \
    --exp1 data/runs/exp1_zero_shot_20260410T.../summary.json \
    --exp2 data/runs/exp2_rag_successful_20260410T.../summary.json \
    --exp3 data/runs/exp3_iql_policy_20260410T.../summary.json
```

---

## Where to Find Outputs

### Training artifacts

| Artifact | Path |
|---|---|
| Trained Q-network | `data/iql/selector/iql_model.pt` |
| Trained V-network | `data/iql/selector/value_model.pt` |
| Normalisation stats | `data/iql/selector/norm_stats.npz` |
| Training loss curves | `iql/plots/training_curves.png` |
| Per-policy Q-values | `iql/plots/per_policy_qvalues.png` |
| IQL dataset | `data/iql/iql_dataset.jsonl` |
| Label map | `data/iql/label_map.json` |
| IQL config | `data/iql/config.json` |
| Operator prototypes | `data/indexes/policies/operator_prototypes.npy` |
| Per-policy pairs | `data/indexes/policies/<resident>_pairs.json` |
| Per-policy FAISS | `data/indexes/faiss/<resident>.faiss` |
| Successful-ops corpus | `data/successful_ops/utterances.jsonl` |
| Successful-ops FAISS | `data/successful_ops/index.faiss` |

### Experiment results

```
data/runs/
├── exp1_zero_shot_<timestamp>/summary.json
├── exp2_rag_successful_<timestamp>/summary.json
├── exp3_iql_policy_<timestamp>/summary.json
├── run_<run_id>/dialogue_<resident>_<run_id>.jsonl
├── comparison.csv
└── matrices_<timestamp>/
    ├── zero_shot_matrix.csv / .png
    ├── rag_successful_matrix.csv / .png
    └── iql_policy_matrix.csv / .png
```

---

## IQL Dueling-Embed Architecture

`I03_train_iql.py` trains the Q-network using a **Dueling embedding-based** architecture designed to produce *distinct* Q-values per policy.

### State representation

The conversation state `s` is the **mean sentence embedding** (via `all-MiniLM-L6-v2`, 384-dim) of the last N resident utterances (`N_LAST_RESIDENT_TRAIN=3` during dataset construction, `N_LAST_RESIDENT_INFER=1` at runtime). This captures what the resident has been saying recently — their resistance level, concerns, and emotional tone — without any manual feature engineering. States are z-normalised using training-set statistics saved to `norm_stats.npz`.

### Policy (action) representations

Each of the 5 operator policies (bob, niki, ross, michelle, lindsay) is represented by a **384-dim embedding vector** — the centroid of all operator utterances in that policy's corpus (the "prototype"). These embeddings are initialised from the pre-computed prototypes but are **trainable** (`nn.Parameter`), allowing the network to push them apart during training. An **orthogonality regularisation** term (`IQL_ORTHO_WEIGHT=0.05`) prevents them from collapsing into each other.

### Why Dueling + Dot-Product (not concatenation)

The original embed architecture concatenated `[state ‖ action_emb]` and passed it through a shared MLP. This consistently produced collapsed Q-values (spread ~0.002) because the network learned to ignore the action embedding portion of the input — the state alone was sufficient to minimise the Bellman loss, so the 384 action-embedding dimensions became dead weight.

The **Dueling architecture** fixes this structurally:

1. **Shared encoder** processes only the state, producing hidden features `h`.
2. **Value head** computes `V(s) = linear(h)` — a scalar baseline.
3. **Advantage head** projects `h` into action-embedding space via a learned linear layer, then computes `A(s,a) = project(h) · embed(a)` via **dot product** with each trainable action embedding.
4. Advantages are centred: `A(s,a) ← A(s,a) − mean_a[A(s,a)]`.
5. Final: `Q(s,a) = V(s) + A(s,a)`.

Because the dot product is computed against geometrically distinct embeddings (maintained by orthogonality regularisation), different actions are **structurally guaranteed** to produce different advantage values, resulting in well-separated Q-values.

### Q-network architecture

```
State (384-dim)
  └─→ Encoder (3 layers):
        Linear(384 → 512) + ReLU + LayerNorm + Dropout(0.55)
        Linear(512 → 512) + ReLU + LayerNorm + Dropout(0.55)
        Linear(512 → 256) + ReLU + LayerNorm + Dropout(0.55)
        → hidden features h (256-dim)
  └─→ Value head:    Linear(256 → 1)              → V(s)    scalar
  └─→ Advantage head: Linear(256 → 384)            → projection (384-dim)
                       projection · action_embeds.T → A(s,a)  (5 values)
                       A ← A − mean(A)             → centred advantages

  Q(s,a) = V(s) + A(s,a)   →  5 Q-values per state
```

A separate **ValueNetwork** `V(s)` (256-dim hidden, 2-layer MLP with LayerNorm + Dropout) is trained in parallel for the Bellman target. A target copy of the V-network is maintained via **exponential moving average** (EMA, `τ=0.005`).

### Training objective

For each `(state s, policy a, reward r, next_state s')` tuple:

```
Bellman target:  y           = r + γ · V_target(s')           [γ = 0.30]
Q-loss:          L_Q         = MSE( Q(s,a),  y )
V-loss:          L_V         = MSE( V(s),    Q(s,a).detach() )
Margin loss:     L_margin    = w · mean( max(0, margin − spread) )
                   where spread = max_a Q(s,a) − min_a Q(s,a)
Ortho loss:      L_ortho     = w · ||E·Eᵀ − I||²
                   where E = L2-normalised action embeddings
Total loss:      L = L_Q + λ_V · L_V + L_margin + L_ortho
```

The **margin loss** penalises the network when the per-sample Q-value spread falls below a threshold (1.0), encouraging policy differentiation. The **orthogonality loss** prevents action embeddings from converging.

Training also applies **Gaussian noise** (`σ=0.10`) to states for augmentation, **gradient clipping** (max norm 1.0), and **class-balanced sampling** over action IDs.

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `IQL_EPOCHS` | 800 | Max training epochs |
| `IQL_BATCH_SIZE` | 16 | Mini-batch size |
| `IQL_LR_Q` | 1e-4 | Q-network learning rate |
| `IQL_LR_V` | 5e-5 | Value network learning rate |
| `IQL_WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `IQL_VAL_SPLIT` | 0.2 | Validation split fraction |
| `IQL_GAMMA` | 0.30 | Bellman discount factor |
| `IQL_LAMBDA_V` | 0.3 | V-loss weight |
| `IQL_MARGIN` | 1.0 | Minimum desired Q-value spread |
| `IQL_MARGIN_WEIGHT` | 0.3 | Margin loss coefficient |
| `IQL_NOISE_STD` | 0.10 | Gaussian noise injected into states during training |
| `IQL_ORTHO_WEIGHT` | 0.05 | Orthogonality regularisation for action embeddings |
| `IQL_TARGET_TAU` | 0.005 | EMA smoothing for target V-network |
| `IQL_GRAD_CLIP` | 1.0 | Max gradient norm |
| `IQL_EARLY_STOP_PATIENCE` | 80 | Epochs without improvement before stopping |
| `IQL_DROPOUT` | 0.55 | Dropout rate |
| `IQL_HIDDEN_DIM_Q` | 512 | Q-network hidden layer width |
| `IQL_HIDDEN_DIM_V` | 256 | Value network hidden layer width |
| `IQL_INFERENCE_TEMP` | 0.9 | Softmax temperature for policy selection at runtime |

### Inference (at runtime)

1. Embed the last `N_LAST_RESIDENT_INFER` (1) resident utterance → `state_vec` (384-dim)
2. Normalise with saved training mean/std from `norm_stats.npz`
3. Single forward pass: `Q(s) → [Q(s,a1), ..., Q(s,a5)]` — all 5 Q-values at once
4. Apply `softmax(Q / temperature)` and **sample** (not argmax) a policy — this allows exploration when Q-values are close
5. Use that policy's FAISS index for per-policy RAG retrieval

---

## Conversation Engine

The conversation engine (`simulation/conversation_loop.py`) runs a single operator↔resident dialogue from start to finish. This is the shared engine used by all three experiments, the interactive mode, and the API.

### Turn flow

1. **Seed**: Either an operator `seed_text` or a default resident line `"Hello? Who is this?"`.
2. **Alternation**: A boolean `operator_next` flag controls whose turn it is.
3. **Operator turn**: Dispatches to the appropriate strategy in `operator_generator.py`:
   - `zero_shot` — Context-only prompt, no examples.
   - `rag_successful` — Injects up to `K_EXAMPLES` (2) retrieved operator utterances from the global successful-operator corpus.
   - `iql_rag` — Runs `IQLPolicySelector.select_policy()`, retrieves up to `K_EXAMPLES` (2) resident→operator pair examples from the selected policy's FAISS index.
4. **Resident turn**: `generate_resident_reply()` uses the persona prompt from `config/personas.py` to ground the LLM.
5. **Decision check**: After each resident turn, `is_successful_session()` is called.
   - **SUCCESS** → Appends a closing operator reply, then stops.
   - **FAILURE** → Stops immediately (refusal streak or turn limit).
   - **None** → Continue to next turn.
6. **Persist**: Each dialogue is saved as JSONL under `data/runs/run_<run_id>/`.

### Simulation settings

| Setting | Value | Description |
|---|---|---|
| `MAX_TURNS` | 15 | Hard cap on conversation turns |
| `K_EXAMPLES` | 2 | Few-shot examples per operator turn |
| `MAX_REFUSAL_STREAK` | 5 | Consecutive refusals → failure |
| `DEFAULT_TEMPERATURE_OP` | 0.2 | Operator LLM temperature |
| `DEFAULT_TEMPERATURE_RES` | 0.2 | Resident LLM temperature |
| `DEFAULT_MAX_TOKENS_OP` | 64 | Operator max response length |
| `DEFAULT_MAX_TOKENS_RES` | 64 | Resident max response length |

---

## Decision Judge

The success/failure judge (`simulation/decision.py`) combines LLM reasoning with rule-based heuristics.

### How it works

1. **Wait for minimum turns**: No decision before `DECISION_MIN_TURNS` (6) utterances.
2. **Refusal keyword counting**: Scans the last 6 resident turns for refusal keywords (e.g. "not leaving", "refuse", "staying put", "safe here"). If `MAX_REFUSAL_STREAK` (5) or more lines contain a keyword → **FAILURE**.
3. **LLM judge**: Sends the last `DECISION_TAIL_WINDOW` (8) turns to the LLM with a strict classification prompt requiring all three:
   - Resident stated a **specific personal concern**
   - Operator addressed that concern with a **concrete, specific plan** (not generic urgency)
   - Resident **explicitly agreed** to evacuate
4. **Hard gate**: LLM SUCCESS is accepted only if `_resident_explicitly_agreed()` matches regex patterns (e.g. "okay I'll go", "let me get ready", "we'll head out") in the latest resident utterance.
5. **Turn limit**: If utterance count ≥ `MAX_TURNS` → **FAILURE**.

### Decision settings

| Setting | Value | Description |
|---|---|---|
| `DECISION_MIN_TURNS` | 6 | Minimum utterances before judging |
| `DECISION_TAIL_WINDOW` | 8 | Number of recent turns sent to LLM judge |

---

## Switching LLM Providers

### OpenAI (recommended for cluster)

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o"    # or gpt-4o-mini
```

### Ollama (local machine)

```bash
ollama pull llama3
ollama serve                          # keep running in a separate terminal

export LLM_PROVIDER=ollama
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3"
```

### Via `config/settings.py`

Edit `LLM_PROVIDER`, `OPENAI_MODEL`, etc. directly in `config/settings.py`.
Environment variables always override the file values.

---

## Interactive Mode

```bash
source .venv/bin/activate
python simulation/interactive.py
```

Starts a console session where you can participate in the evacuation conversation on either or both sides.

### Modes

| Flag | Description |
|---|---|
| `--role operator` | Human plays operator, AI plays resident (default) |
| `--role resident` | AI plays operator, human plays resident |
| `--role both` | Human plays both sides (full manual) |
| `--role none` | AI plays both sides (automated, same as experiments) |

### Examples

```bash
# Play as operator against AI-simulated Ross
python simulation/interactive.py --resident ross --role operator

# Watch a fully automated IQL conversation with Bob
python simulation/interactive.py --resident bob --role none --strategy iql_rag

# Play as operator with zero-shot strategy
python simulation/interactive.py --resident michelle --role operator --strategy zero_shot
```

When in `--role operator` mode with `--strategy iql_rag`, the IQL recommendation and Q-values are displayed before each operator turn to guide the human.

---

## API Server

```bash
source .venv/bin/activate
python api/server.py
# or: uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/personas` | GET | List available resident persona names |
| `/chat` | POST | Single-turn: send history + latest utterance, get next reply |
| `/simulate` | POST | Full multi-turn automated conversation |

### `/chat` request body

```json
{
  "speaker": "operator",
  "resident": "ross",
  "history": [{"role": "resident", "text": "Hello? Who is this?"}],
  "text": "Hello? Who is this?",
  "strategy": "iql_rag",
  "temperature": 0.7,
  "max_tokens": 128
}
```

### `/simulate` request body

```json
{
  "resident": "ross",
  "strategy": "iql_rag",
  "seed_text": "Hello, this is the fire department...",
  "max_turns": 16
}
```

### Response

`/chat` returns `{"text": "...", "selected_policy": "bob"}`.

`/simulate` returns `{"status": "SUCCESS", "success": 1, "turns": 8, "history": [...], "path": "..."}`.
