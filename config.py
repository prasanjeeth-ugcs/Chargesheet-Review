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
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# ── Document Processing ───────────────────────────────────────────────────────
# Gemini 2.0 Flash supports ~1M tokens. We send the entire (truncated) document
# in a SINGLE API call — no chunking needed. Only 2 API calls total per document:
#   Call 1: Combined Summary + Classification
#   Call 2: Checklist Analysis
CHUNK_SIZE = 8000          # kept for backward compat / notebook use only
CHUNK_OVERLAP = 500        # kept for backward compat
MAX_TEXT_LENGTH = 100000   # max characters to process (~30K tokens, well within 1M limit)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
LLM_CALL_DELAY = 3         # seconds to wait between sequential API calls
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

# ── OCR Correction Mode ───────────────────────────────────────────────────────
# Controls how Hindi OCR errors (ि↔न confusion etc.) are corrected:
#   "pattern"  : General algorithmic pattern rules only (fast, no extra API cost)
#   "llm"      : Use LLM to fix OCR errors (highest accuracy, +1 API call)
#   "hybrid"   : Pattern rules first, then LLM for remaining (best balance)
# For production police deployments, "hybrid" or "llm" recommended.
OCR_CORRECTION_MODE = os.getenv("OCR_CORRECTION_MODE", "pattern")

# ── NER / Timeline Quality Thresholds ───────────────────────────────────────
# Cross-script (Roman ↔ Devanagari) person dedup similarity threshold.
# Higher values are stricter; lower values merge more aggressively.
CROSS_SCRIPT_DEDUP_THRESHOLD = float(os.getenv("CROSS_SCRIPT_DEDUP_THRESHOLD", "0.72"))

# Minimum amount (₹) for regex-based monetary recovery from raw text.
# Helps ignore incidental tiny numbers while keeping case-relevant amounts.
MONETARY_MIN_AMOUNT = float(os.getenv("MONETARY_MIN_AMOUNT", "500"))

# OCR warning threshold shown to users in report summary/output.
# If OCR confidence is below this, UI prepends a high-visibility warning banner.
OCR_LOW_QUALITY_THRESHOLD = float(os.getenv("OCR_LOW_QUALITY_THRESHOLD", "0.70"))

# Context window (characters) for deriving PERSON contextual hints from source text.
PERSON_CONTEXT_WINDOW_CHARS = int(os.getenv("PERSON_CONTEXT_WINDOW_CHARS", "120"))

# Context window (characters) to inspect around each monetary amount candidate.
# Used to suppress routine bank-statement noise amounts (fees/tax/balance rows).
MONETARY_CONTEXT_WINDOW_CHARS = int(os.getenv("MONETARY_CONTEXT_WINDOW_CHARS", "100"))

# Small monetary amounts at or below this value are treated as potential statement-noise
# ONLY when accompanied by fee/charge/tax context words.
MONETARY_NOISE_MAX_AMOUNT = float(os.getenv("MONETARY_NOISE_MAX_AMOUNT", "2000"))

# Wider secondary context window for timeline financial-event detection.
# Helps classify deposit/withdrawal dates correctly even when the primary sentence window is narrow.
FINANCIAL_SECONDARY_WINDOW_CHARS = int(os.getenv("FINANCIAL_SECONDARY_WINDOW_CHARS", "400"))

# EVIDENCE dedup tolerance (character edit distance) for OCR/cross-script variants.
# Example merges: "Blood Sample" vs OCR-broken transliteration variants.
EVIDENCE_DEDUP_EDIT_DISTANCE = int(os.getenv("EVIDENCE_DEDUP_EDIT_DISTANCE", "3"))

# Minimum length for substring-based EVIDENCE dedup safety guard.
EVIDENCE_DEDUP_MIN_LENGTH = int(os.getenv("EVIDENCE_DEDUP_MIN_LENGTH", "5"))
