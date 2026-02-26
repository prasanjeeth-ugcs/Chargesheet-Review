"""
Configuration for Smart Chargesheet Review & Summarisation Assistant.
Supports Google Gemini (free tier) and OpenAI GPT backends.
"""

import os

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Set to "gemini" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Multiple Gemini API keys for rotation (comma-separated)
# Each free-tier key has its own quota, so rotating keys multiplies capacity
GEMINI_API_KEYS: list[str] = []   # populated at runtime from UI or env
_env_keys = os.getenv("GEMINI_API_KEYS", "")
if _env_keys:
    GEMINI_API_KEYS = [k.strip() for k in _env_keys.split(",") if k.strip()]

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Document Processing ───────────────────────────────────────────────────────
# Gemini 2.0 Flash supports ~1M tokens. We send the entire (truncated) document
# in a SINGLE API call — no chunking needed. Only 2 API calls total per document:
#   Call 1: Combined Summary + Classification
#   Call 2: Checklist Analysis
CHUNK_SIZE = 8000          # kept for backward compat / notebook use only
CHUNK_OVERLAP = 500        # kept for backward compat
MAX_TEXT_LENGTH = 100000   # max characters to process (~30K tokens, well within 1M limit)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
LLM_CALL_DELAY = 5         # seconds to wait between the 2 API calls
LLM_MAX_RETRIES = 3        # max retries on rate-limit (429) errors per model
LLM_RETRY_BASE_DELAY = 15  # base delay in seconds for exponential backoff

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKLIST_PATH = os.path.join(os.path.dirname(__file__), "checklists.json")

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = "Smart Chargesheet Review & Summarisation Assistant"
APP_DESCRIPTION = (
    "Upload a Hindi chargesheet (text / .docx / .pdf) to get a structured summary, "
    "crime-type classification, and a checklist of missing documents."
)
