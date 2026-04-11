"""
settings.py — Central Configuration
=====================================

Single source of truth for every path, hyperparameter, device, and service
endpoint used across the project.

All other modules import from here — nothing is hard-coded elsewhere.

To switch the LLM backend (e.g. from Ollama/llama3 to OpenAI GPT-4):
    1. Set LLM_PROVIDER = "openai"
    2. Set OPENAI_API_KEY (env var or directly)
    3. Set LLM_MODEL to the desired model name (e.g. "gpt-4o")
"""

import os
import torch
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT ROOT
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PATHS (all relative to PROJECT_ROOT)
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR        = PROJECT_ROOT / "data"
RAW_XLSX_DIR    = DATA_DIR / "raw_xlsx"
JSONL_DIR       = DATA_DIR / "jsonl"
CLEANED_DIR     = DATA_DIR / "cleaned"
META_DIR        = DATA_DIR / "meta"
REPORTS_DIR     = DATA_DIR / "reports"

IQL_DIR         = DATA_DIR / "iql"
SELECTOR_DIR    = IQL_DIR / "selector"

# Training plots go here — inside the iql/ source folder so they are
# committed to the repository alongside the code.
IQL_PLOTS_DIR   = PROJECT_ROOT / "iql" / "plots"

INDEXES_DIR     = DATA_DIR / "indexes"
POLICY_DIR      = INDEXES_DIR / "policies"
FAISS_DIR       = INDEXES_DIR / "faiss"

SUCCESSFUL_OPS_DIR = DATA_DIR / "successful_ops"
RUNS_DIR           = DATA_DIR / "runs"

RESIDENTS_META_FILE   = META_DIR / "residents.json"
IQL_DATASET_FILE      = IQL_DIR / "iql_dataset.jsonl"
IQL_CONFIG_FILE       = IQL_DIR / "config.json"
LABEL_MAP_FILE        = IQL_DIR / "label_map.json"
PROTOTYPES_FILE       = POLICY_DIR / "operator_prototypes.npy"
POLICIES_META_FILE    = POLICY_DIR / "policies_meta.json"
FAISS_META_FILE       = FAISS_DIR / "meta_faiss.json"

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE — prefer cuda > mps > cpu
# ═══════════════════════════════════════════════════════════════════════════════
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL (sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════════
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Supported providers: "ollama", "openai"
# To add a new provider, implement a handler in simulation/llm_client.py
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# --- Ollama settings ---------------------------------------------------------
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# --- OpenAI settings ---------------------------------------------------------
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o")

# --- Resolved model name (used by llm_client) --------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", OLLAMA_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL)

# ═══════════════════════════════════════════════════════════════════════════════
# IQL TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
IQL_EPOCHS             = 2000
IQL_BATCH_SIZE         = 32
IQL_LR_Q               = 3e-6
IQL_LR_V               = 3e-4
IQL_VAL_SPLIT          = 0.2
IQL_GAMMA              = 0.50
IQL_LAMBDA_V           = 0.3
IQL_EARLY_STOP_PATIENCE = 100
IQL_DROPOUT            = 0.3
IQL_HIDDEN_DIM_Q       = 1024
IQL_HIDDEN_DIM_V       = 512

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
N_LAST_RESIDENT_TRAIN = 4   # window used when building the IQL dataset (A04)
N_LAST_RESIDENT_INFER = 3   # window used at runtime for policy selection

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
MAX_TURNS         = 16
K_EXAMPLES        = 2     # ICL few-shot examples per operator turn
MAX_REFUSAL_STREAK = 5    # consecutive refusals → stop

DEFAULT_TEMPERATURE_OP  = 0.5
DEFAULT_TEMPERATURE_RES = 0.8
DEFAULT_MAX_TOKENS_OP   = 32
DEFAULT_MAX_TOKENS_RES  = 32

# ═══════════════════════════════════════════════════════════════════════════════
# DECISION JUDGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DECISION_MIN_TURNS   = 4
DECISION_TAIL_WINDOW = 4

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN HINTS (for XLSX parsing)
# ═══════════════════════════════════════════════════════════════════════════════
DROP_SPEAKERS = ["julie", "system", "narrator"]

RESIDENTS_LIST = ["bob", "michelle", "ross", "niki", "lindsay"]

# ═══════════════════════════════════════════════════════════════════════════════
# API SERVER
# ═══════════════════════════════════════════════════════════════════════════════
API_HOST = "0.0.0.0"
API_PORT = 8001

# ═══════════════════════════════════════════════════════════════════════════════
# Ensure required directories exist
# ═══════════════════════════════════════════════════════════════════════════════
for _d in [
    RAW_XLSX_DIR, JSONL_DIR, CLEANED_DIR, META_DIR, REPORTS_DIR,
    IQL_DIR, SELECTOR_DIR, POLICY_DIR, FAISS_DIR,
    SUCCESSFUL_OPS_DIR, RUNS_DIR, IQL_PLOTS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)
