"""
Core processing logic for the Smart Chargesheet Review & Summarisation Assistant.

OPTIMISED for minimal API calls:
  - Call 1: Combined Summary + Crime Classification (single LLM call)
  - Call 2: Checklist analysis (single LLM call)
  Total: only 2 API calls per document (Gemini supports 1M tokens natively)

Also includes:
  - Text extraction from .txt / .docx / .pdf
  - Smart text truncation for very large docs
  - Retry with exponential backoff for rate limits
  - Model fallback chain (tries multiple models)
  - Rule-based keyword detection as cross-validation
"""

import json
import os
import re
import time
import logging
import base64
import io
import math
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(file_path: str) -> str:
    """Extract plain text from .txt, .docx, or .pdf files."""
    text, _ = extract_text_and_images(file_path)
    return text


def extract_text_and_images(file_path: str) -> tuple:
    """
    Extract plain text AND embedded images from .txt, .docx, or .pdf files.
    Returns (text: str, images: list[bytes])  where images are raw PNG/JPEG bytes.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), []

    elif ext == ".docx":
        return _extract_from_docx(file_path)

    elif ext == ".pdf":
        return _extract_from_pdf(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt, .docx, or .pdf")


def _extract_from_docx(file_path: str) -> tuple:
    """Extract text + images from .docx file, with [PAGE N] markers."""
    try:
        import docx
        from docx.oxml.ns import qn
    except ImportError:
        raise ImportError("python-docx is required. Install: pip install python-docx")

    doc = docx.Document(file_path)

    # Extract text with page markers
    page_num = 1
    text_parts = ["\n[PAGE 1]\n"]

    for paragraph in doc.paragraphs:
        # Detect page breaks (explicit breaks + rendered page breaks)
        has_page_break = False
        for run in paragraph.runs:
            # Explicit page break: <w:br w:type="page"/>
            for br in run._element.findall(qn('w:br')):
                if br.get(qn('w:type')) == 'page':
                    has_page_break = True
            # Rendered page break: <w:lastRenderedPageBreak/>
            for _ in run._element.iter(qn('w:lastRenderedPageBreak')):
                has_page_break = True

        if has_page_break:
            page_num += 1
            text_parts.append(f"\n[PAGE {page_num}]\n")

        text_parts.append(paragraph.text)

    text = "\n".join(text_parts)

    # Also extract text from tables (case diaries often use tables)
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                text += "\n" + row_text

    # Extract embedded images
    images = []
    try:
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_data = rel.target_part.blob
                    if image_data and len(image_data) > 1000:  # skip tiny icons
                        images.append(image_data)
                except Exception as e:
                    logger.warning(f"Could not extract image: {e}")
    except Exception as e:
        logger.warning(f"Could not access docx images: {e}")

    logger.info(f"DOCX extracted: {len(text):,} chars text, {len(images)} images")

    # Compress large images to keep API payload reasonable
    images = _compress_images(images)

    return text, images


def _extract_from_pdf(file_path: str) -> tuple:
    """Extract text + images from .pdf file, with [PAGE N] markers."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required. Install: pip install PyPDF2")

    text_parts = []
    images = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            text_parts.append(f"\n[PAGE {page_num}]\n")
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

            # Extract images from PDF pages
            try:
                if hasattr(page, 'images'):
                    for img in page.images:
                        if hasattr(img, 'data') and len(img.data) > 1000:
                            images.append(img.data)
            except Exception as e:
                logger.debug(f"Could not extract PDF image: {e}")

    text = "\n".join(text_parts)
    logger.info(f"PDF extracted: {len(text):,} chars text, {len(images)} images")

    # Compress large images
    images = _compress_images(images)

    return text, images


def _compress_images(images: list, max_size: int = 1_500_000, max_dim: int = 2048) -> list:
    """
    Compress images to keep API payload reasonable.
    - Resizes images larger than max_dim pixels on any side
    - Re-encodes to JPEG if larger than max_size bytes
    - Limits total to 20 images
    """
    if not images:
        return images

    compressed = []
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        logger.warning("Pillow not installed — images will be sent without compression. Install: pip install Pillow")

    for i, img_bytes in enumerate(images[:20]):
        if has_pil and len(img_bytes) > max_size:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                # Resize if too large
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                # Re-encode as JPEG
                buf = io.BytesIO()
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(buf, format='JPEG', quality=80)
                compressed_bytes = buf.getvalue()
                logger.debug(f"Image {i+1}: {len(img_bytes):,} -> {len(compressed_bytes):,} bytes")
                compressed.append(compressed_bytes)
            except Exception as e:
                logger.warning(f"Could not compress image {i+1}: {e}, using original")
                compressed.append(img_bytes)
        else:
            compressed.append(img_bytes)

    total_size = sum(len(img) for img in compressed)
    logger.info(f"Images ready: {len(compressed)} images, total {total_size:,} bytes")
    return compressed


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Preparation (Smart Truncation)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 2b. Text Preprocessing (OCR cleanup, noise reduction)
# ─────────────────────────────────────────────────────────────────────────────

# Common OCR garbage patterns
_OCR_GARBAGE_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f]'   # control characters
    r'|\ufffd'                             # unicode replacement char
    r'|[\u200b-\u200f\u202a-\u202e]'     # zero-width / bidi chars
    r'|\u00a0{3,}'                         # runs of non-breaking spaces
)

# Broken Devanagari: isolated matras without a base consonant
_BROKEN_DEVANAGARI_RE = re.compile(
    r'(?<![\u0915-\u0939\u0958-\u0961])'  # not preceded by a consonant
    r'[\u093e-\u094d\u0951-\u0954]{2,}'   # 2+ consecutive matras/nukta/virama
)

# Excessive punctuation / repeated symbols
_EXCESSIVE_PUNCT_RE = re.compile(r'([\-_=*#~.]{4,})')

# Whitespace normalizer
_MULTI_SPACE_RE = re.compile(r'[^\S\n]{2,}')  # 2+ spaces (not newlines)
_MULTI_NEWLINE_RE = re.compile(r'\n{4,}')       # 4+ blank lines → 2
_TRAILING_SPACE_RE = re.compile(r'[ \t]+$', re.MULTILINE)


def preprocess_text(text: str) -> str:
    """
    Clean raw OCR / chargesheet text before sending to the LLM.
    Steps:
      1. Strip control characters and Unicode noise
      2. Remove broken Devanagari matra sequences
      3. Collapse excessive punctuation / decorative lines
      4. Normalize whitespace (multiple spaces, blank lines)
      5. Normalize common Hindi date formats
      6. Strip page headers/footers that repeat across pages
    Does NOT alter actual content — only removes noise.
    """
    if not text:
        return text

    original_len = len(text)

    # 1. Remove control chars and Unicode noise
    text = _OCR_GARBAGE_RE.sub('', text)

    # 2. Remove broken Devanagari sequences (isolated matras)
    text = _BROKEN_DEVANAGARI_RE.sub('', text)

    # 3. Collapse excessive punctuation / decorative lines
    text = _EXCESSIVE_PUNCT_RE.sub(lambda m: m.group(1)[:3], text)

    # 4. Normalize whitespace
    text = _MULTI_SPACE_RE.sub(' ', text)
    text = _TRAILING_SPACE_RE.sub('', text)
    text = _MULTI_NEWLINE_RE.sub('\n\n', text)

    # 5. Normalize common Hindi date formats: 01.02.2024 → 01/02/2024
    text = re.sub(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', r'\1/\2/\3', text)

    # 6. Remove repeated page headers/footers (lines appearing 3+ times exactly)
    lines = text.split('\n')
    if len(lines) > 20:
        line_counts = {}
        for line in lines:
            stripped = line.strip()
            if len(stripped) > 5:  # ignore very short lines
                line_counts[stripped] = line_counts.get(stripped, 0) + 1
        repeated = {l for l, c in line_counts.items() if c >= 3}
        if repeated:
            lines = [l for l in lines if l.strip() not in repeated]
            text = '\n'.join(lines)

    text = text.strip()

    cleaned_len = len(text)
    if original_len - cleaned_len > 100:
        logger.info(f"Preprocessing: cleaned {original_len:,} → {cleaned_len:,} chars "
                    f"(removed {original_len - cleaned_len:,} noise chars)")

    return text


def _truncate_text(text: str) -> str:
    """
    Truncate text to MAX_TEXT_LENGTH if too long.
    Keeps beginning (case header, FIR) and end (conclusion, sections, attachments).
    Gemini 2.0 Flash supports ~1M tokens, so 80K chars (~25K tokens) is very safe.
    """
    max_len = getattr(config, 'MAX_TEXT_LENGTH', 80000)
    if len(text) <= max_len:
        return text

    # Keep 60% from start (FIR, parties, incident), 40% from end (sections, conclusion)
    start_len = int(max_len * 0.6)
    end_len = max_len - start_len

    truncated = (
        text[:start_len]
        + "\n\n[... दस्तावेज़ का मध्य भाग छोड़ा गया / middle portion omitted ...]\n\n"
        + text[-end_len:]
    )
    logger.info(f"Document truncated: {len(text):,} -> {len(truncated):,} characters")
    return truncated


# ─────────────────────────────────────────────────────────────────────────────
# 3. Checklists
# ─────────────────────────────────────────────────────────────────────────────

def load_checklists() -> dict:
    """Load crime-type checklists from JSON."""
    with open(config.CHECKLIST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_crime_type_info(crime_key: str) -> dict | None:
    """Return checklist info for a specific crime key."""
    checklists = load_checklists()
    return checklists.get(crime_key)


def list_crime_types() -> list[dict]:
    """Return a list of available crime types with display names."""
    checklists = load_checklists()
    return [
        {"key": k, "display_name": v["display_name"], "display_name_en": v["display_name_en"]}
        for k, v in checklists.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM Interaction (with retry + model fallback)
# ─────────────────────────────────────────────────────────────────────────────

# Models to try in order if one hits rate limits or is unavailable
GEMINI_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-05-20",
    "gemini-1.5-flash",
]

# Cache for discovered available models
_available_models_cache = None

# Track which API key index to use next (round-robin)
_current_key_index = 0

# Remember which model last succeeded — start with it on next call
_last_successful_model = None


def _discover_available_models(client) -> list:
    """Discover which Gemini models are actually available for this API key."""
    global _available_models_cache
    if _available_models_cache is not None:
        return _available_models_cache

    provider_type, client_obj = client
    if provider_type != "gemini":
        return GEMINI_FALLBACK_MODELS

    try:
        all_models = list(client_obj.models.list())
        flash_models = []
        for m in all_models:
            name = m.name if hasattr(m, 'name') else str(m)
            # Strip "models/" prefix if present
            short_name = name.replace("models/", "")
            if "flash" in short_name.lower():
                flash_models.append(short_name)
        if flash_models:
            logger.info(f"Available flash models: {flash_models}")
            _available_models_cache = flash_models
            return flash_models
    except Exception as e:
        logger.warning(f"Could not discover models: {e}")

    return GEMINI_FALLBACK_MODELS


def _get_all_api_keys() -> list:
    """Get all available Gemini API keys (multi-key pool)."""
    keys = list(config.GEMINI_API_KEYS) if config.GEMINI_API_KEYS else []
    # Always include the primary key if set and not already in list
    if config.GEMINI_API_KEY and config.GEMINI_API_KEY not in keys:
        keys.insert(0, config.GEMINI_API_KEY)
    return keys


def _get_gemini_client(api_key: str = None):
    """Get Gemini client with a specific API key."""
    key = api_key or config.GEMINI_API_KEY
    try:
        from google import genai
        client = genai.Client(
            api_key=key,
            http_options={'timeout': 300_000},  # 300 seconds in ms
        )
        return ("gemini", client)
    except ImportError:
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=key)
            return ("gemini_old", genai_old)
        except ImportError:
            raise ImportError("google-genai is required. Install: pip install google-genai")


def _get_openai_client():
    """Get OpenAI client."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        return ("openai", client)
    except ImportError:
        raise ImportError("openai is required. Install: pip install openai")


def _call_gemini(client, model_name: str, prompt, images: list = None) -> str:
    """
    Call Gemini API with specific model.
    Supports multimodal: if images are provided, sends text + images together.
    """
    provider_type, client_obj = client

    if provider_type == "gemini":
        from google.genai import types

        # Build multimodal content if images are present
        if images:
            contents = []
            # Add text prompt first
            contents.append(types.Part.from_text(text=prompt if isinstance(prompt, str) else prompt))
            # Add images (limit to 20 to stay within API limits)
            for i, img_bytes in enumerate(images[:20]):
                try:
                    # Detect mime type
                    mime = "image/jpeg"
                    if img_bytes[:4] == b'\x89PNG':
                        mime = "image/png"
                    elif img_bytes[:4] == b'GIF8':
                        mime = "image/gif"
                    elif img_bytes[:2] == b'BM':
                        mime = "image/bmp"

                    contents.append(types.Part.from_bytes(
                        data=img_bytes,
                        mime_type=mime,
                    ))
                except Exception as e:
                    logger.warning(f"Could not add image {i+1}: {e}")

            logger.info(f"Sending multimodal request: text + {len([c for c in contents if hasattr(c, 'inline_data')])} images")
            response = client_obj.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=65536,
                ),
            )
        else:
            response = client_obj.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=65536,
                ),
            )
        return response.text
    elif provider_type == "gemini_old":
        model = client_obj.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0},
        )
        return response.text


def _call_llm(prompt: str, system_prompt: str = "", images: list = None) -> str:
    """
    Send a prompt to the LLM with:
      - Retry with exponential backoff on 429 errors
      - Model fallback chain (tries alternate models if one is rate-limited)
      - Multimodal support: pass images for Gemini vision
    """
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        return _call_openai(full_prompt, system_prompt, prompt)

    # Gemini path: rotate across multiple API keys + model fallback
    global _current_key_index, _last_successful_model
    # Only 1 retry per key — fail fast and move to next key/model
    max_retries = 1
    base_delay = 5  # seconds

    api_keys = _get_all_api_keys()
    if not api_keys:
        raise ValueError("No Gemini API key provided. Please enter your API key.")

    configured_model = config.GEMINI_MODEL

    # Discover available models using first key
    first_client = _get_gemini_client(api_keys[0])
    available = _discover_available_models(first_client)

    # Build a SHORT, sensible model list (not all 18 discovered models)
    # Priority: last successful > configured > curated fallbacks
    preferred_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
    ]
    # Only keep models that actually exist in the API
    models_to_try = []
    for m in preferred_models:
        if m in available or m in GEMINI_FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)
    # Ensure configured model is in the list
    if configured_model not in models_to_try:
        models_to_try.insert(0, configured_model)

    # If a model succeeded recently, put it FIRST
    if _last_successful_model and _last_successful_model in models_to_try:
        models_to_try.remove(_last_successful_model)
        models_to_try.insert(0, _last_successful_model)

    num_keys = len(api_keys)
    logger.info(f"API keys available: {num_keys}, Models to try: {models_to_try}")

    last_error = None

    # Strategy: for each model, rotate through ALL keys before moving to next model
    for model_name in models_to_try:
        for key_offset in range(num_keys):
            key_idx = (_current_key_index + key_offset) % num_keys
            api_key = api_keys[key_idx]
            key_label = f"key{key_idx+1}/{num_keys}"

            client = _get_gemini_client(api_key)

            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Calling {model_name} with {key_label} (attempt {attempt+1})")
                    result = _call_gemini(client, model_name, full_prompt, images=images)
                    # Success! Remember this model + advance key index
                    _last_successful_model = model_name
                    _current_key_index = (key_idx + 1) % num_keys
                    logger.info(f"Success with {model_name} {key_label}")
                    return result
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    is_rate_limit = any(kw in error_str for kw in [
                        "429", "RESOURCE_EXHAUSTED", "rate_limit", "quota",
                        "Too Many Requests", "RateLimitError",
                    ])
                    is_timeout = any(kw in error_str for kw in [
                        "timed out", "timeout", "TimeoutError",
                        "DeadlineExceeded", "DEADLINE_EXCEEDED",
                    ])
                    is_model_not_found = any(kw in error_str for kw in [
                        "404", "NOT_FOUND", "not found", "not supported",
                        "does not exist", "is not available",
                    ])

                    if is_model_not_found:
                        logger.warning(f"Model {model_name} not available, trying next model...")
                        break  # break attempt loop, will also break key loop below
                    elif is_timeout:
                        logger.warning(f"Timeout on {model_name} {key_label}, trying next key...")
                        break  # skip to next key
                    elif is_rate_limit and attempt < max_retries:
                        wait_time = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Rate limit on {model_name} {key_label} (attempt {attempt+1}). "
                            f"Waiting {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    elif is_rate_limit:
                        # This key is exhausted, try next key
                        logger.warning(f"Key {key_label} exhausted for {model_name}, trying next key...")
                        break
                    else:
                        raise
            else:
                # attempt loop completed without break — shouldn't happen, but continue
                continue

            # If we broke out due to model-not-found, skip remaining keys for this model
            if last_error and any(kw in str(last_error) for kw in ["404", "NOT_FOUND", "not found", "not supported"]):
                break

    # All keys, models and retries exhausted
    raise last_error


def _call_openai(full_prompt: str, system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI API."""
    _, client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINED Summary + Classification (SINGLE API CALL)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a legal document analysis assistant specialising in Indian police chargesheets. "
    "You can read and understand Hindi and English. Always respond in a structured format. "
    "Be precise, factual, and do not hallucinate information not present in the document."
)


def _build_combined_prompt(text: str) -> str:
    """Build a single prompt that asks for BOTH summary AND crime classification."""
    checklists = load_checklists()
    crime_types_str = "\n".join(
        f"- **{k}**: {v['display_name']} (Sections: {', '.join(v['typical_sections'])})"
        for k, v in checklists.items()
    )

    return f"""निम्नलिखित हिन्दी चार्जशीट/आरोप पत्र का विश्लेषण करें। आपको दो कार्य करने हैं:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
कार्य 1: संरचित सारांश (Structured Summary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

निम्न प्रारूप में सारांश दें:

## केस हेडर (Case Header)
- **FIR संख्या (FIR Number):**
- **दिनांक (Date):**
- **थाना (Police Station):**
- **न्यायालय (Court):**
- **घटना स्थल (Place of Occurrence):**
- **घटना दिनांक एवं समय (Date & Time of Incident):**

## पक्षकार (Parties Involved)
### शिकायतकर्ता / पीड़ित (Complainant / Victim):
- नाम, पिता का नाम, आयु, पता

### आरोपी (Accused):
- नाम, पिता का नाम, आयु, पता, उपनाम (यदि कोई हो)

### मुख्य गवाह (Key Witnesses):
- गवाहों की सूची (नाम व संक्षिप्त विवरण)

## घटना का सारांश (Incident Summary)
(क्या हुआ, कैसे, कब, कहाँ – 5-10 पंक्तियों में)

## लागू कानूनी धाराएं (Legal Sections Applied)
- धाराओं की सूची (IPC / अन्य अधिनियम) – हिन्दी और अंग्रेजी दोनों में

## प्रमुख साक्ष्य (Key Evidence)
- बरामद/जब्त वस्तुओं की सूची
- रिपोर्ट/दस्तावेज जो चार्जशीट में उल्लिखित हैं

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
कार्य 2: अपराध वर्गीकरण (Crime Classification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

उपलब्ध अपराध श्रेणियाँ:
{crime_types_str}

सारांश के बाद, निम्न JSON ब्लॉक दें (```json ... ``` में):
```json
{{{{
  "primary_crime_type": "<crime_key>",
  "secondary_crime_types": ["<crime_key>", ...],
  "detected_sections": ["IPC 392", "IPC 323", ...],
  "confidence": "high/medium/low",
  "confidence_score": 0.85,
  "reasoning": "संक्षिप्त कारण (2-3 पंक्तियाँ)"
}}}}
```

confidence_score के लिए:
- 0.85–1.0: धाराएँ स्पष्ट मैच, कोई अंबिगुइटी नहीं
- 0.6–0.84: धाराएँ आंशिक मैच / अनुमानित
- 0.3–0.59: अस्पष्ट, मिश्रित संकेत
- 0.0–0.29: अनिश्चित / अनुमान

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
कार्य 3: Named Entity Recognition (NER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

दस्तावेज़ से सभी महत्वपूर्ण NAMED ENTITIES निकालें।
Classification JSON के बाद, एक और JSON ब्लॉक दें:

```ner_json
{{{{
  "entities": [
    {{{{"text": "entity का exact नाम", "type": "ENTITY_TYPE"}}}},
    ...
  ]
}}}}
```

Entity Types (role-based person classification):
- ACCUSED: आरोपी / अभियुक्त का proper name (e.g. "मो0 सादिक", "रमन कुमार")
- WITNESS: गवाह / साक्षी का proper name (e.g. "सुनील यादव", "राजू प्रसाद")
- OFFICER: पुलिस/सरकारी अधिकारी — IO, SI, ASI, SP, थानाध्यक्ष (e.g. "अभिषेक कुमार", "राजेश सिंह")
- DOCTOR: चिकित्सक/डॉक्टर — जिन्होंने मेडिकल जाँच या पोस्टमार्टम किया (e.g. "डॉ. विभाकर कुमार")
- PERSON: कोई अन्य व्यक्ति — पीड़ित, शिकायतकर्ता, रिश्तेदार, अन्य उल्लिखित (e.g. "रमेश कुमार", "सीमा देवी")

Non-person entity types:
- DATE: हर तिथि अलग-अलग निकालें — FIR date, घटना date, गिरफ्तारी date, मेडिकल date, सब अलग हैं। कोई भी date merge/skip न करें। (e.g. "15/03/2024", "दिनांक 20.04.2024", "01.01.2025")
- LOCATION: स्थान/शहर/गाँव/मोहल्ला (e.g. "चन्द्रपुर", "बॉण्डी बस्ती")
- LEGAL_SECTION: कानूनी धारा (e.g. "BNS 103(1)", "Arms Act 27")
- ORGANIZATION: थाना/न्यायालय/अस्पताल/FSL/सरकारी संस्था (e.g. "सरायकेला थाना", "TMH")
- LANDMARK: दुकान/कम्पनी/मंदिर — स्थान-पहचान के लिए (e.g. "प्रतिक्षा टेक्सटाईल")
- EVIDENCE: जब्त/बरामद भौतिक वस्तु (e.g. "पिस्टल .315 बोर", "खून के नमूने")
- MONETARY: धनराशि (e.g. "₹50,000", "1,20,000 रुपये")

⚠️ DATE extraction — विशेष निर्देश:
- दस्तावेज़ में हर अलग-अलग date को एक अलग entity के रूप में निकालें
- FIR date, घटना date, गिरफ्तारी date, जमानत date, मेडिकल date, रिपोर्ट date — सब अलग-अलग entries होनी चाहिए
- अपेक्षित: 20-40 DATE entities (timeline reconstruction के लिए)
- Duplicate dates (exact same string) हटाएं, लेकिन अलग-अलग dates कभी merge न करें

⚠️ सख्त NER नियम (STRICT RULES — अवश्य पालन करें):

1. **केवल NAMED ENTITIES** — proper nouns, specific names, specific dates, specific amounts
2. **"text" में केवल entity का नाम लिखें** — कोई भूमिका, सम्बन्ध, विवरण, या व्याख्या नहीं
   - ✅ {{{{"text": "मो0 आजाद", "type": "PERSON"}}}}
   - ❌ {{{{"text": "मो0 आजाद — मो0 इरफान के जीजा", "type": "PERSON"}}}}
   - ❌ {{{{"text": "मो0 आजाद (accused)", "type": "PERSON"}}}}
3. **सम्बन्ध-वाक्य entity नहीं हैं** — "जियाउल की बिवी", "आरोपी का भाई" = NOT entities
4. **एक व्यक्ति = एक entry** — यदि नाम कई बार आता है तो केवल एक बार लिखें
5. **OCR variations को एक ही entity मानें** — "मो0 सादिक" और "मो0 सादीक" = same person, केवल एक बार लिखें
6. **LEGAL_SECTION: केवल इस FIR/चार्जशीट में लगाई गई धाराएँ** — पिछले अपराधों की धाराएँ शामिल न करें
7. **EVIDENCE: केवल भौतिक वस्तुएँ** — "witness statement" evidence नहीं है
8. **कुल entities 100-150 के बीच होनी चाहिए** (dates 20-40 अलग + persons 30-50 + rest)
9. **Role classification**: ACCUSED/WITNESS/OFFICER/DOCTOR/PERSON — context से तय करें, न कि अनुमान से
- ORGANIZATION: थाना, न्यायालय, अस्पताल, FSL, सरकारी संस्था
- LANDMARK: दुकान, कम्पनी, मंदिर, मस्जिद, चौराहा — जो स्थान-पहचान के लिए उल्लेख हुए

- यदि कोई entity नहीं मिली तो खाली सूची [] रखें
- सभी entities हिन्दी में लिखें

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
चार्जशीट दस्तावेज़ (Chargesheet Document):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{text}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 5b. NER Post-Processing & Validation
# ─────────────────────────────────────────────────────────────────────────────

# Hindi relationship words — if entity text contains these, it's NOT a named entity
_RELATION_WORDS = re.compile(
    r'\b(की\s+बिवी|का\s+भाई|की\s+बहन|का\s+बेटा|की\s+बेटी|का\s+पिता|की\s+माता|'
    r'का\s+पति|की\s+पत्नी|का\s+ससुर|की\s+सास|का\s+दामाद|की\s+बहू|'
    r'का\s+चाचा|की\s+चाची|का\s+मामा|की\s+मामी|का\s+साला|की\s+साली|'
    r'का\s+जीजा|की\s+जीजी|का\s+देवर|की\s+देवरानी|का\s+भतीजा|की\s+भतीजी|'
    r'के\s+रिश्तेदार|का\s+रिश्तेदार)\b',
    re.IGNORECASE
)

# IPC sections — old law (should not appear if FIR uses BNS)
_IPC_SECTIONS = re.compile(r'\b(?:IPC|आईपीसी|भारतीय\s+दण्ड\s+संहिता)\s*(?:धारा\s*)?\d+', re.IGNORECASE)
# BNS/BNSS sections — new law
_BNS_SECTIONS = re.compile(r'\b(?:BNS|BNSS|बीएनएस|भारतीय\s+न्याय\s+संहिता|भारतीय\s+नागरिक\s+सुरक्षा\s+संहिता)\s*(?:धारा\s*)?\d+', re.IGNORECASE)


def _normalize_entity_text(text: str, entity_type: str = "") -> str:
    """Normalize entity text for deduplication comparison.
    For DATE entities, only normalize whitespace (preserve exact date strings).
    For person-like entities, normalize Hindi honorifics.
    """
    t = text.strip()

    if entity_type == "DATE":
        # Minimal normalization for dates — only whitespace & separators
        t = re.sub(r'\s+', ' ', t).strip()
        # Normalize date separators: 15-03-2024, 15/03/2024, 15.03.2024
        t = re.sub(r'[\-/\.]', '/', t)
        return t.lower()

    # Person/general normalization
    # Normalize Hindi honorifics and abbreviations
    t = re.sub(r'मो[0०\.]+\s*', 'मो. ', t)
    t = re.sub(r'डॉ[0०\.]+\s*', 'डॉ. ', t)
    t = re.sub(r'Dr\.?\s*', 'डॉ. ', t, flags=re.IGNORECASE)
    t = re.sub(r'श्री[\s\.]*', '', t)
    t = re.sub(r'Sri\.?\s*|Shri\.?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'कु[0०\.]+\s*', 'कु. ', t)
    t = re.sub(r'स[0०]+\s*अ[0०]+\s*नि[0०]+\s*', '', t)  # Remove स0 अ0 नि0
    t = re.sub(r'Mohd?\.?\s*', 'मो. ', t, flags=re.IGNORECASE)
    # Normalize spaces
    t = re.sub(r'\s+', ' ', t).strip()
    # Lowercase for comparison
    return t.lower()


def _strip_annotations(text: str) -> str:
    """
    Strip role/relation annotations from entity text.
    'मो0 आजाद — मो0 इरफान के जीजा' → 'मो0 आजाद'
    'राकेश कुमार (witness)' → 'राकेश कुमार'
    'अभिषेक कुमार (IO, सरायकेला थाना)' → 'अभिषेक कुमार'
    """
    # Remove everything after em-dash or double-hyphen
    text = re.split(r'\s*[—–\-]{1,2}\s+', text, maxsplit=1)[0].strip()
    # Remove parenthetical annotations
    text = re.sub(r'\s*\([^)]*\)\s*$', '', text).strip()
    # Remove trailing comma + role
    text = re.sub(r',\s*(?:accused|victim|witness|complainant|IO|SI|ASI|आरोपी|पीड़ित|गवाह|शिकायतकर्ता).*$',
                  '', text, flags=re.IGNORECASE).strip()
    return text


def _is_relation_phrase(text: str) -> bool:
    """Check if entity text is actually a relationship phrase, not a named entity."""
    if _RELATION_WORDS.search(text):
        return True
    # Check for patterns like "X के Y" where Y is a relation word
    if re.search(r'\b(?:के|की|का)\s+(?:बिवी|पत्नी|पति|भाई|बहन|बेटा|बेटी|पिता|माता|'
                 r'ससुर|सास|जीजा|देवर|चाचा|मामा|साला|भतीजा|रिश्तेदार)', text):
        return True
    return False


# Person-like types that need relation-phrase rejection
_PERSON_TYPES = {"PERSON", "ACCUSED", "WITNESS", "OFFICER", "DOCTOR"}


def _canonicalize_entities(entities: list) -> list:
    """
    Deduplicate entities using normalized text comparison.
    Key design:
    - Normalize BEFORE clustering (not after)
    - DATE entities: exact-match dedup only (no fuzzy) — preserve timeline
    - Person entities: fuzzy dedup with 80% similarity threshold
    - Person-like types dedup across roles (same person can't be ACCUSED + PERSON)
    Returns list of unique entity dicts.
    """
    if not entities:
        return entities

    # Step 1: Normalize all entity text FIRST (before any grouping/clustering)
    for ent in entities:
        raw = ent.get("text", "").strip()
        ent["_raw"] = raw
        # Strip annotations before normalization
        clean = _strip_annotations(raw)
        ent["text"] = clean if clean else raw
        ent["_norm"] = _normalize_entity_text(clean if clean else raw, ent.get("type", ""))

    # Step 2: Dedup DATE entities separately (exact match only — no fuzzy)
    date_entities = [e for e in entities if e.get("type") == "DATE"]
    other_entities = [e for e in entities if e.get("type") != "DATE"]

    deduped_dates = []
    seen_date_norms = set()
    for ent in date_entities:
        norm = ent["_norm"]
        if not ent["text"] or not norm:
            continue
        if norm not in seen_date_norms:
            seen_date_norms.add(norm)
            deduped_dates.append({"text": ent["text"], "type": "DATE"})

    # Step 3: Dedup person-like entities across all person roles together
    #   (prevents "मो0 समीर" appearing as both ACCUSED and PERSON)
    person_entities = [e for e in other_entities if e.get("type") in _PERSON_TYPES]
    non_person_entities = [e for e in other_entities if e.get("type") not in _PERSON_TYPES]

    deduped_persons = []
    seen_person_norms = {}  # norm → canonical entry
    for ent in person_entities:
        raw_text = ent.get("_raw", "")
        clean_text = ent["text"]
        norm = ent["_norm"]
        etype = ent.get("type", "PERSON")

        if not clean_text:
            continue

        # Reject relation phrases
        if _is_relation_phrase(raw_text):
            logger.info(f"NER dedup: Rejected relation phrase: '{raw_text}'")
            continue

        # Fuzzy match across all person types
        matched = False
        for seen_norm, canonical in seen_person_norms.items():
            if _fuzzy_match(norm, seen_norm):
                # Keep longer form; keep more specific role
                if len(clean_text) > len(canonical["text"]):
                    canonical["text"] = clean_text
                # Prefer specific role over generic PERSON
                if canonical["type"] == "PERSON" and etype in ("ACCUSED", "WITNESS", "OFFICER", "DOCTOR"):
                    canonical["type"] = etype
                matched = True
                break

        if not matched:
            entry = {"text": clean_text, "type": etype}
            seen_person_norms[norm] = entry
            deduped_persons.append(entry)

    # Step 4: Dedup non-person, non-date entities (by type, with fuzzy match)
    by_type = {}
    for ent in non_person_entities:
        etype = ent.get("type", "UNKNOWN")
        by_type.setdefault(etype, []).append(ent)

    deduped_other = []
    for etype, items in by_type.items():
        seen_normalized = {}
        for item in items:
            clean_text = item["text"]
            norm = item["_norm"]
            if not clean_text:
                continue

            matched = False
            for seen_norm, canonical in seen_normalized.items():
                if _fuzzy_match(norm, seen_norm):
                    if len(clean_text) > len(canonical["text"]):
                        canonical["text"] = clean_text
                    matched = True
                    break

            if not matched:
                entry = {"text": clean_text, "type": etype}
                seen_normalized[norm] = entry
                deduped_other.append(entry)

    # Combine all
    result = deduped_persons + deduped_dates + deduped_other

    # Clean up internal keys
    for ent in result:
        ent.pop("_raw", None)
        ent.pop("_norm", None)

    return result


def _fuzzy_match(a: str, b: str) -> bool:
    """
    Check if two normalized entity texts are likely the same entity.
    Uses character-level edit distance ratio for Hindi OCR variation tolerance.
    """
    if a == b:
        return True
    if not a or not b:
        return False

    # If one is a substring of the other (e.g. 'सादिक' in 'मो. सादिक')
    if a in b or b in a:
        return True

    # Character-level similarity (Levenshtein-like)
    # For short names, even 1-2 char difference in Hindi can be OCR noise
    longer = max(len(a), len(b))
    if longer == 0:
        return True

    # Simple edit distance (optimized for short strings)
    dist = _edit_distance(a, b)
    ratio = 1.0 - (dist / longer)

    # Threshold: 80% similarity for strings >= 4 chars, exact match for shorter
    if longer < 4:
        return a == b
    return ratio >= 0.80


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _convert_ner_to_flat_list(ner: dict) -> list:
    """
    Convert old-format NER dict {"TYPE": ["str", ...]} to new flat list format
    [{"text": "...", "type": "TYPE"}, ...].
    Handles both old dict-of-lists format and new list-of-dicts format.
    """
    if isinstance(ner, list):
        return ner  # already in new format

    entities = []
    for etype, items in ner.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                entities.append(item)
            elif isinstance(item, str):
                entities.append({"text": item, "type": etype})
    return entities


def _validate_legal_sections(entities: list, classification: dict) -> list:
    """
    Validate LEGAL_SECTION entities: prevent cross-contamination
    between BNS (new law) and IPC (old law).
    """
    legal_entities = [e for e in entities if e.get("type") == "LEGAL_SECTION"]
    other_entities = [e for e in entities if e.get("type") != "LEGAL_SECTION"]

    if not legal_entities:
        return entities

    all_legal_text = " ".join(e["text"] for e in legal_entities)
    has_bns = bool(_BNS_SECTIONS.search(all_legal_text))
    has_ipc = bool(_IPC_SECTIONS.search(all_legal_text))

    # Also check classification
    detected = classification.get("detected_sections", [])
    detected_text = " ".join(str(s) for s in detected) if detected else ""
    if _BNS_SECTIONS.search(detected_text):
        has_bns = True
    if _IPC_SECTIONS.search(detected_text):
        has_ipc = True

    if has_bns:
        # Filter out IPC sections from a BNS FIR
        filtered_legal = []
        removed = 0
        for e in legal_entities:
            if _IPC_SECTIONS.search(e["text"]) and not _BNS_SECTIONS.search(e["text"]):
                logger.info(f"NER validation: Removed hallucinated IPC section: {e['text']}")
                removed += 1
                continue
            filtered_legal.append(e)
        if removed:
            logger.info(f"NER validation: Removed {removed} IPC sections from BNS FIR")
        return other_entities + filtered_legal

    return entities


def _process_ner_output(raw_ner: dict | list, classification: dict) -> list:
    """
    Full NER post-processing pipeline:
    1. Convert to flat list format
    2. Strip annotations from entity text
    3. Reject relation phrases
    4. Canonicalize / deduplicate
    5. Validate legal sections
    Returns clean list of {"text": ..., "type": ...} dicts.
    """
    # Step 1: Flatten
    entities = _convert_ner_to_flat_list(raw_ner)

    # Step 2-4: Canonicalize (normalize first, then cluster + dedup + relation rejection)
    entities = _canonicalize_entities(entities)

    # Step 5: Legal section validation
    entities = _validate_legal_sections(entities, classification)

    logger.info(f"NER post-processing: {len(entities)} canonical entities")
    return entities


def _parse_combined_response(response: str) -> tuple:
    """
    Parse the combined response into summary, classification, and NER.
    Returns (summary, classification, ner_entities).
    """
    # --- Parse Classification JSON ---
    json_match = re.search(r'```json\s*([\s\S]*?)```', response)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*"primary_crime_type"[^{}]*\})', response, re.DOTALL)

    classification = {}
    summary = response

    if json_match:
        json_str = json_match.group(1) if '```' in response else json_match.group(0)
        try:
            classification = json.loads(json_str.strip())
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r',\s*}', '}', json_str.strip())
                cleaned = re.sub(r',\s*]', ']', cleaned)
                classification = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning("Could not parse classification JSON from response")

        # Remove Task 2 and Task 3 sections from summary display
        task2_pattern = re.search(r'━+\s*\n\s*कार्य 2', summary)

        if task2_pattern:
            summary = summary[:task2_pattern.start()].strip()
        else:
            summary = re.sub(r'```json[\s\S]*?```', '', summary)
            summary = re.sub(r'```ner_json[\s\S]*?```', '', summary).strip()

    if not classification:
        classification = {
            "primary_crime_type": "unknown",
            "secondary_crime_types": [],
            "detected_sections": [],
            "confidence": "low",
            "reasoning": "Could not parse classification from response",
        }

    # --- Parse NER JSON ---
    ner_entities = {}
    ner_match = re.search(r'```ner_json\s*([\s\S]*?)```', response)
    if not ner_match:
        # Fallback: look for a JSON block containing "entities"
        ner_match = re.search(r'(\{[\s\S]*?"entities"[\s\S]*?\})\s*(?:```|$)', response)
    logger.info(f"NER block search: found={'YES' if ner_match else 'NO'}, response length={len(response)}")
    if ner_match:
        ner_str = ner_match.group(1).strip()
        logger.info(f"NER JSON block found ({len(ner_str)} chars)")
        try:
            ner_data = json.loads(ner_str)
            ner_entities = ner_data.get("entities", ner_data)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r',\s*}', '}', ner_str)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                ner_data = json.loads(cleaned)
                ner_entities = ner_data.get("entities", ner_data)
            except json.JSONDecodeError:
                logger.warning("Could not parse NER JSON from response")
    else:
        logger.warning("No NER JSON block found in response (looked for ```ner_json)")

    # --- Full NER post-processing: flatten → strip → dedup → validate ---
    ner_entities = _process_ner_output(ner_entities, classification)

    return summary, classification, ner_entities


# ─────────────────────────────────────────────────────────────────────────────
# 6. Checklist Analysis (SECOND API CALL)
# ─────────────────────────────────────────────────────────────────────────────

def _build_checklist_prompt(text: str, crime_key: str, required_items: list) -> str:
    items_str = "\n".join(
        f"  {i+1}. [{item['id']}] {item['label_hi']} ({item['label_en']})"
        for i, item in enumerate(required_items)
    )

    return f"""निम्नलिखित चार्जशीट दस्तावेज़ का विश्लेषण करें और बताएं कि नीचे दी गई आवश्यक वस्तुओं/दस्तावेजों में से कौन-कौन दस्तावेज़ में उल्लिखित/उपस्थित हैं और कौन-कौन अनुपस्थित या अपूर्ण हैं।

**अपराध श्रेणी:** {crime_key}

**आवश्यक वस्तुओं/दस्तावेजों की सूची:**
{items_str}

**निर्देश:**
- प्रत्येक वस्तु के लिए बताएं: "present" (उपस्थित), "missing" (अनुपस्थित), या "partial" (आंशिक/अपूर्ण)।
- यदि "partial" है तो क्या कमी है, बताएं।
- दस्तावेज़ में [PAGE N] मार्कर दिए गए हैं। प्रत्येक "present" या "partial" वस्तु के लिए "page_no" फ़ील्ड में बताएं कि यह जानकारी किस पेज पर मिली (e.g. "3" or "5-7")। यदि "missing" है तो page_no खाली रखें ""।
- JSON format में उत्तर दें।

**आउटपुट format:**
```json
{{{{
  "checklist": [
    {{{{"id": "<item_id>", "status": "present|missing|partial", "page_no": "<page number or range>", "remarks": "टिप्पणी"}}}},
    ...
  ]
}}}}
```

---
**चार्जशीट दस्तावेज़:**
{text}
"""


def analyse_checklist(text: str, crime_key: str) -> dict:
    """
    Analyse chargesheet against the required-items checklist.
    Uses LLM for accuracy, with rule-based fallback.
    """
    crime_info = get_crime_type_info(crime_key)
    if not crime_info:
        return {"error": f"Unknown crime type: {crime_key}"}

    required_items = crime_info["required_items"]
    prompt = _build_checklist_prompt(text, crime_key, required_items)

    try:
        response = _call_llm(prompt, SYSTEM_PROMPT)

        # Extract JSON
        json_match = re.search(r'```json\s*([\s\S]*?)```', response)
        if not json_match:
            json_match = re.search(r'\{[\s\S]*"checklist"[\s\S]*\}', response)

        if json_match:
            json_str = json_match.group(1) if '```json' in response else json_match.group(0)
            try:
                result = json.loads(json_str.strip())
                return result
            except json.JSONDecodeError:
                pass

        # If JSON parsing failed, try rule-based
        logger.warning("LLM checklist response not parseable, using rule-based fallback")
        return _rule_based_checklist(text, required_items)

    except Exception as e:
        # LLM failed entirely, use rule-based
        logger.warning(f"LLM checklist call failed ({e}), using rule-based fallback")
        return _rule_based_checklist(text, required_items)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rule-based Keyword Detection (cross-validation, no API)
# ─────────────────────────────────────────────────────────────────────────────

def detect_crime_type_rules(text: str) -> list:
    """
    Rule-based crime type detection using keyword matching.
    Returns list of matching crime types sorted by score.
    No API calls -- runs entirely offline.
    """
    checklists = load_checklists()
    text_lower = text.lower()
    results = []

    for key, info in checklists.items():
        score = 0

        # Check for legal section mentions
        for section in info["typical_sections"]:
            section_num = section.split()[-1]
            section_patterns = [
                section.lower(),
                f"धारा {section_num}",
                f"section {section_num}",
            ]
            for pat in section_patterns:
                if pat.lower() in text_lower:
                    score += 3

        # Check Hindi keywords
        for kw in info.get("keywords_hi", []):
            if kw in text:
                score += 2

        # Check English keywords
        for kw in info.get("keywords_en", []):
            if kw.lower() in text_lower:
                score += 1

        if score > 0:
            results.append({
                "crime_key": key,
                "display_name": info["display_name"],
                "score": score,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _rule_based_checklist(text: str, required_items: list) -> dict:
    """Fallback rule-based checklist using keyword matching."""
    text_lower = text.lower()
    checklist = []

    keyword_map = {
        # === Shared / Common ===
        "fir": ["FIR", "प्राथमिकी", "प्रथम सूचना रिपोर्ट", "एफआईआर"],
        "ps": ["थाना", "पुलिस स्टेशन", "police station"],
        "place_time": ["घटनास्थल", "स्थान", "समय", "बजे", "दिनांक"],
        "complainant": ["शिकायतकर्ता", "पीड़ित", "complainant", "victim"],
        "accused": ["आरोपी", "अभियुक्त", "accused", "संदिग्ध"],
        "witnesses": ["गवाह", "प्रत्यक्षदर्शी", "पंच गवाह", "witness", "eye-witness"],
        "witness_stmt": ["गवाह", "बयान", "witness", "statement", "161"],
        "arrest_memo": ["गिरफ्तारी मेमो", "arrest memo", "गिरफ्तार"],

        # === Theft / Robbery ===
        "property_desc": ["संपत्ति", "सामान", "चोरी", "stolen property", "मोबाइल", "मूल्य"],
        "ownership_proof": ["स्वामित्व", "बिल", "ownership", "receipt"],
        "seizure_memo": ["जब्ती मेमो", "seizure memo", "बरामदगी मेमो", "बरामद"],
        "chain_custody": ["कस्टडी चेन", "custody", "chain of custody"],
        "site_plan": ["नक्शा मौका", "site plan", "स्थल निरीक्षण", "घटना स्थल का नक्शा"],
        "cctv": ["CCTV", "सीसीटीवी", "कैमरा", "फुटेज"],
        "spot_photos": ["फोटोग्राफ", "तस्वीर", "photograph", "फोटो"],
        "confession_disclosure": ["स्वीकारोक्ति", "प्रकटीकरण", "confession", "disclosure"],
        "personal_search": ["व्यक्तिगत तलाशी", "personal search"],
        "property_seal": ["सील", "seal", "जब्ती मेमो"],
        "malkhana_forward": ["मालखाना", "malkhana", "अग्रेषण"],
        "supplementary": ["पूरक आरोप पत्र", "supplementary"],

        # === Assault / Hurt ===
        "victim": ["पीड़ित", "victim", "शिकायतकर्ता"],
        "mlc": ["MLC", "मेडिको लीगल", "चिकित्सा परीक्षण", "medical"],
        "injury_cert": ["चोट प्रमाणपत्र", "injury certificate", "चोटों की प्रकृति"],
        "injury_photos": ["चोटों के फोटो", "तस्वीर", "photograph", "इंजरी फोटो"],
        "doctor_opinion": ["डॉक्टर की राय", "doctor's opinion", "चिकित्सक", "साधारण", "गम्भीर"],
        "referral_docs": ["रेफरल", "referral", "उच्च केंद्र"],
        "weapon_desc": ["हथियार", "चाकू", "लाठी", "डंडी", "बंदूक", "weapon", "knife", "lathi"],
        "weapon_seizure": ["हथियार", "जब्ती", "weapon seizure"],
        "motive": ["मकसद", "मोटिव", "motive", "रंजिश", "enmity", "कारण"],
        "section_mapping": ["धारा-वार", "section-wise", "section mapping"],
        "arrest_medical": ["गिरफ्तारी", "आरोपी का चिकित्सा", "accused medical"],

        # === Cyber Fraud ===
        "platform_mode": ["प्लेटफॉर्म", "ऑनलाइन", "platform", "UPI", "बैंकिंग ऐप", "OLX", "सोशल मीडिया"],
        "txn_datetime": ["लेनदेन", "ट्रांजेक्शन", "transaction", "दिनांक"],
        "amount": ["राशि", "amount", "रुपये", "₹"],
        "txn_records": ["बैंक स्टेटमेंट", "transaction record", "UPI", "ट्रांजेक्शन रिकॉर्ड"],
        "payment_logs": ["पेमेंट गेटवे", "payment gateway", "लॉग"],
        "chat_logs": ["चैट", "ईमेल", "screenshot", "स्क्रीनशॉट", "कॉल डिटेल", "CDR"],
        "email_headers": ["ईमेल हेडर", "email header", "मैसेज ID"],
        "device_ids": ["IMEI", "IP", "डिवाइस", "यूजर ID", "device identifier"],
        "device_seizure": ["डिजिटल", "मोबाइल", "लैपटॉप", "device", "जब्ती"],
        "forensic_image": ["फोरेंसिक इमेज", "forensic image", "हैश", "hash"],
        "bank_request": ["बैंक", "KYC", "अनुरोध", "bank request"],
        "bank_response": ["बैंक से प्राप्त", "bank response", "उत्तर", "response"],
        "digital_id_link": ["IP", "IMEI", "digital identifier", "पहचान से जोड़"],
        "chain_digital": ["डिजिटल साक्ष्य", "digital evidence", "कस्टडी"],
        "fsl_cyber": ["साइबर फोरेंसिक", "cyber forensic", "FSL"],
        "account_freeze": ["खाता फ्रीज", "account freeze", "रिकवरी"],

        # === NDPS ===
        "seizure_place_time": ["जब्ती का स्थान", "seizure place", "नाका"],
        "substance_desc": ["पदार्थ", "मात्रा", "substance", "quantity", "किलोग्राम", "ग्राम", "पाउडर"],
        "ndps_compliance": ["धारा 42", "धारा 50", "NDPS", "अनुपालन"],
        "search_memo": ["तलाशी", "search memo", "तलाशी मेमो"],
        "weighment": ["तौल", "weighment", "वजन", "पंचनामा"],
        "sample_memo": ["नमूना", "sample", "ड्राइंग मेमो"],
        "malkhana": ["मालखाना", "malkhana", "रजिस्टर"],
        "fsl_dispatch": ["FSL भेजने", "dispatch", "FSL memo"],
        "fsl_report": ["FSL", "फॉरेंसिक", "forensic", "एफएसएल"],
        "expert_opinion": ["विशेषज्ञ राय", "expert opinion"],
        "independent_witness": ["स्वतंत्र गवाह", "independent witness", "पंच"],
        "mandatory_compliance": ["अनिवार्य प्रावधान", "mandatory provision", "अनुपालन"],
    }

    for item in required_items:
        item_id = item["id"]
        keywords = keyword_map.get(item_id, [])
        found = any(kw.lower() in text_lower or kw in text for kw in keywords)
        checklist.append({
            "id": item_id,
            "status": "present" if found else "missing",
            "remarks": "",
        })

    return {"checklist": checklist}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Semantic Similarity Scoring (Stage 2B)
# ─────────────────────────────────────────────────────────────────────────────

# Threshold for considering a checklist item as semantically PRESENT
SEMANTIC_THRESHOLD = 0.25  # TF-IDF cosine on Hindi text — threshold is lower than English


def _split_into_sentences(text: str) -> list:
    """
    Split Hindi/English text into sentence-like units.
    Uses punctuation and newline boundaries.
    """
    # Split on Hindi purna viram, question marks, newlines, and semicolons
    raw = re.split(r'[।\.\?!;\n]+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 10:  # skip very short fragments
            sentences.append(s)
    return sentences


def compute_semantic_similarity(
    text: str,
    required_items: list,
) -> dict:
    """
    Stage 2B: Semantic Similarity Scoring.
    Uses TF-IDF + cosine similarity to match each required checklist item
    against document sentences. This catches paraphrased mentions that
    keyword matching would miss (e.g. "पंचनामा तैयार किया गया" for "Weighment panchnama").

    Returns dict:
    {
        "item_id": {
            "similarity_score": float,      # 0.0 to 1.0
            "semantic_status": "present" | "missing" | "partial",
            "best_match": "matching sentence from document",
            "confidence": "high" | "medium" | "low"
        },
        ...
    }
    """
    sentences = _split_into_sentences(text)
    if not sentences or not required_items:
        return {}

    # Build query strings from checklist items (combine Hindi + English labels + keywords)
    queries = []
    item_ids = []
    for item in required_items:
        # Combine all available text representations for richer matching
        parts = [item["label_hi"], item["label_en"]]
        # Add any keywords if available
        if "keywords" in item:
            parts.extend(item["keywords"])
        queries.append(" ".join(parts))
        item_ids.append(item["id"])

    # Combine queries + sentences for TF-IDF fitting
    all_texts = queries + sentences

    try:
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',   # char n-grams work better for Hindi/mixed text
            ngram_range=(2, 4),   # 2-4 char grams capture Hindi morphology
            max_features=15000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Split: first N rows are queries, rest are sentences
        query_vectors = tfidf_matrix[:len(queries)]
        sentence_vectors = tfidf_matrix[len(queries):]

        # Compute cosine similarity: queries × sentences
        sim_matrix = cosine_similarity(query_vectors, sentence_vectors)

    except Exception as e:
        logger.warning(f"Semantic similarity computation failed: {e}")
        return {}

    results = {}
    for i, item_id in enumerate(item_ids):
        scores = sim_matrix[i]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_sentence = sentences[best_idx] if best_idx < len(sentences) else ""

        # Truncate long matching sentences
        if len(best_sentence) > 200:
            best_sentence = best_sentence[:200] + "…"

        # Determine status and confidence
        if best_score >= SEMANTIC_THRESHOLD * 1.5:  # strong match
            status = "present"
            confidence = "high"
        elif best_score >= SEMANTIC_THRESHOLD:       # moderate match
            status = "partial"
            confidence = "medium"
        else:
            status = "missing"
            confidence = "low"

        results[item_id] = {
            "similarity_score": round(best_score, 3),
            "semantic_status": status,
            "best_match": best_sentence,
            "confidence": confidence,
        }

    return results


def _merge_checklist_with_rules(
    llm_result: dict,
    rule_result: dict,
) -> dict:
    """
    Merge LLM checklist with rule-based keyword matching (OR logic).
    If LLM says MISSING but rule-based found keywords → upgrade to PRESENT.
    If LLM says PARTIAL but rule-based found keywords → upgrade to PRESENT.
    """
    llm_checklist = llm_result.get("checklist", [])
    rule_checklist = rule_result.get("checklist", [])

    if not llm_checklist or not rule_checklist:
        return llm_result

    # Build rule lookup: item_id → status
    rule_lookup = {}
    for entry in rule_checklist:
        rule_lookup[entry.get("id", "")] = entry.get("status", "missing")

    merged = []
    for entry in llm_checklist:
        new_entry = dict(entry)
        item_id = entry.get("id", "")
        rule_status = rule_lookup.get(item_id, "missing")

        # Upgrade logic: rule says present → override LLM missing/partial
        if entry.get("status") in ("missing", "partial") and rule_status == "present":
            old_status = entry.get("status")
            new_entry["status"] = "present"
            if not new_entry.get("remarks"):
                new_entry["remarks"] = "🔑 Keyword match found in document"
            else:
                new_entry["remarks"] += " | 🔑 Keyword match"
            logger.info(f"Rule upgrade: {item_id} {old_status.upper()}→PRESENT (keyword match)")

        merged.append(new_entry)

    return {"checklist": merged}


def _merge_checklist_with_similarity(
    checklist_result: dict,
    similarity_scores: dict,
) -> dict:
    """
    Merge LLM checklist results with semantic similarity scores.
    The LLM result is primary; similarity scores augment it with:
    - A similarity confidence score
    - The best matching sentence
    - Upgrade MISSING → PARTIAL if semantic match found
    """
    checklist = checklist_result.get("checklist", [])
    if not checklist or not similarity_scores:
        return checklist_result

    merged = []
    for entry in checklist:
        item_id = entry.get("id", "")
        sim_info = similarity_scores.get(item_id, {})

        # Copy existing entry
        new_entry = dict(entry)

        if sim_info:
            new_entry["similarity_score"] = sim_info["similarity_score"]
            new_entry["best_match"] = sim_info["best_match"]

            # Upgrade: if LLM says MISSING/PARTIAL but semantic says PRESENT
            if entry.get("status") in ("missing", "partial") and sim_info["semantic_status"] == "present":
                new_entry["status"] = "present"
                if not new_entry.get("remarks"):
                    new_entry["remarks"] = "🔍 Strong semantic match found"
                else:
                    new_entry["remarks"] += " | 🔍 Strong semantic match"
                logger.info(f"Semantic upgrade: {item_id} {entry.get('status').upper()}→PRESENT (score={sim_info['similarity_score']:.3f})")
            elif entry.get("status") == "missing" and sim_info["semantic_status"] == "partial":
                new_entry["status"] = "partial"
                if not new_entry.get("remarks"):
                    new_entry["remarks"] = "🔍 Partial semantic match found"
                else:
                    new_entry["remarks"] += " | 🔍 Partial semantic match"
                logger.info(f"Semantic upgrade: {item_id} MISSING→PARTIAL (score={sim_info['similarity_score']:.3f})")

        merged.append(new_entry)

    return {"checklist": merged}


# ─────────────────────────────────────────────────────────────────────────────
# 8b. Chunking (kept for backward compatibility but NOT used in main pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list:
    """Split text into overlapping chunks. Kept for notebook/advanced usage."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 8c. Confidence Scoring (Stage 3A — computed, not random)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_classification_confidence(
    llm_classification: dict,
    rule_classification: list,
) -> float:
    """
    Compute composite classification confidence (0.0–1.0).
    Triangulates 2 independent signals:
    - LLM confidence (high/medium/low → 0.9/0.6/0.3)  — weight 0.55
    - LLM confidence_score (if provided)               — override LLM weight
    - Rule-based score (normalized)                     — weight 0.45

    Returns a meaningful 0.0–1.0 score.
    """
    # Signal 1: LLM-declared confidence
    llm_conf_str = llm_classification.get("confidence", "low")
    llm_score_raw = llm_classification.get("confidence_score", None)
    if llm_score_raw is not None:
        try:
            llm_signal = float(llm_score_raw)
            llm_signal = max(0.0, min(1.0, llm_signal))
        except (ValueError, TypeError):
            llm_signal = {"high": 0.90, "medium": 0.60, "low": 0.30}.get(llm_conf_str, 0.30)
    else:
        llm_signal = {"high": 0.90, "medium": 0.60, "low": 0.30}.get(llm_conf_str, 0.30)

    # Signal 2: Rule-based section/keyword score
    primary = llm_classification.get("primary_crime_type", "unknown")
    rule_score = 0.0
    if rule_classification:
        # Find matching rule result for the primary crime
        for r in rule_classification:
            if r["crime_key"] == primary:
                # Normalize: typical score range 5-30, map to 0-1
                rule_score = min(1.0, r["score"] / 20.0)
                break
        if rule_score == 0.0:
            # Primary didn't match any rule → penalty
            rule_score = 0.1

    # Weighted combination
    composite = (0.55 * llm_signal) + (0.45 * rule_score)
    composite = round(max(0.0, min(1.0, composite)), 2)

    logger.info(f"Classification confidence: LLM={llm_signal:.2f} Rule={rule_score:.2f} "
                f"→ Composite={composite:.2f}")

    return composite


def _compute_checklist_item_confidence(entry: dict) -> float:
    """
    Compute per-item confidence (0.0–1.0) for a checklist item.
    Triangulates:
    - LLM status (present=0.70, partial=0.40, missing=0.05)  — base
    - Rule-based keyword match (if remarks mention 🔑)       — +0.15
    - Semantic similarity score (if available)                — +0.15 × score
    """
    status = entry.get("status", "missing")
    remarks = entry.get("remarks", "") or ""
    sim_score = entry.get("similarity_score", 0.0) or 0.0

    # Base confidence from LLM status
    if status == "present":
        base = 0.70
    elif status == "partial":
        base = 0.40
    else:
        base = 0.05

    # Rule-based bonus
    rule_bonus = 0.15 if "🔑" in remarks else 0.0

    # Semantic bonus (scaled)
    sem_bonus = 0.15 * min(1.0, sim_score / 0.4) if sim_score > 0 else 0.0

    # Strong semantic match bonus
    if "🔍 Strong semantic" in remarks:
        sem_bonus = max(sem_bonus, 0.12)
    elif "🔍 Partial semantic" in remarks:
        sem_bonus = max(sem_bonus, 0.06)

    confidence = round(min(1.0, base + rule_bonus + sem_bonus), 2)
    return confidence


def _enrich_checklist_with_confidence(checklist_result: dict) -> dict:
    """Add per-item confidence scores to every checklist entry."""
    checklist = checklist_result.get("checklist", [])
    for entry in checklist:
        entry["confidence"] = _compute_checklist_item_confidence(entry)
    return checklist_result


def _compute_field_confidence(summary: str) -> dict:
    """
    Heuristic field confidence: check which fields are present in the summary.
    Scans for field headers and content to determine 0.0–1.0 confidence.
    No API call needed — pure text analysis.
    """
    # Field detection patterns: (field_key, [header_patterns], [content_patterns])
    field_checks = {
        "fir_number": {
            "headers": [r"FIR\s*(?:संख्या|नं|No)", r"प्रथम सूचना रिपोर्ट", r"काण्ड\s*संख्या"],
            "content": [r"\d{2,6}\s*/\s*\d{4}", r"FIR\s*(?:No|नं)\.?\s*\d+"],
        },
        "fir_date": {
            "headers": [r"FIR\s*(?:दिनांक|Date|तिथि)", r"दिनांक"],
            "content": [r"\d{1,2}[/\.\-]\d{1,2}[/\.\-]\d{2,4}"],
        },
        "police_station": {
            "headers": [r"थाना", r"Police\s*Station"],
            "content": [r"थाना\s*[:\-]?\s*\S+"],
        },
        "court": {
            "headers": [r"न्यायालय", r"Court", r"माननीय"],
            "content": [r"न्यायालय\s*[:\-]?\s*\S+", r"Court\s*[:\-]?\s*\S+"],
        },
        "place_of_occurrence": {
            "headers": [r"घटना\s*स्थल", r"Place\s*of\s*Occurrence", r"घटनास्थल"],
            "content": [r"घटना\s*स्थल\s*[:\-]?\s*\S+"],
        },
        "incident_date_time": {
            "headers": [r"घटना\s*(?:दिनांक|तिथि|समय|Date)", r"Date.*Time.*Incident"],
            "content": [r"घटना\s*(?:दिनांक|तिथि)\s*[:\-]?\s*\d"],
        },
        "complainant_name": {
            "headers": [r"शिकायतकर्ता", r"पीड़ित", r"Complainant", r"Victim"],
            "content": [r"शिकायतकर्ता\s*[:\-]?\s*\S+", r"पीड़ित\s*[:\-]?\s*\S+"],
        },
        "accused_names": {
            "headers": [r"आरोपी", r"अभियुक्त", r"Accused"],
            "content": [r"आरोपी\s*[:\-]?\s*\S+", r"अभियुक्त\s*[:\-]?\s*\S+"],
        },
        "witnesses": {
            "headers": [r"गवाह", r"साक्षी", r"Witness"],
            "content": [r"गवाह\s*[:\-]?\s*\S+", r"साक्षी\s*[:\-]?\s*\S+"],
        },
        "incident_summary": {
            "headers": [r"घटना\s*का\s*सारांश", r"Incident\s*Summary", r"सारांश"],
            "content": [r"घटना\s*का\s*सारांश"],
        },
        "legal_sections": {
            "headers": [r"कानूनी\s*धारा", r"Legal\s*Section", r"लागू.*धारा"],
            "content": [r"धारा\s*\d+", r"BNS\s*\d+", r"IPC\s*\d+"],
        },
        "key_evidence": {
            "headers": [r"प्रमुख\s*साक्ष्य", r"Key\s*Evidence", r"साक्ष्य"],
            "content": [r"प्रमुख\s*साक्ष्य", r"बरामद", r"जब्त"],
        },
    }

    result = {}
    for field_key, checks in field_checks.items():
        header_found = False
        content_found = False

        for pat in checks["headers"]:
            if re.search(pat, summary, re.IGNORECASE):
                header_found = True
                break

        for pat in checks["content"]:
            if re.search(pat, summary, re.IGNORECASE):
                content_found = True
                break

        if header_found and content_found:
            result[field_key] = 0.90
        elif header_found:
            result[field_key] = 0.60  # Header present but no clear content
        elif content_found:
            result[field_key] = 0.70  # Content detected without clear header
        else:
            result[field_key] = 0.10  # Not found

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full Pipeline -- OPTIMISED: only 2 API calls
# ─────────────────────────────────────────────────────────────────────────────

def process_chargesheet(
    text: str,
    manual_crime_type: Optional[str] = None,
    images: list = None,
) -> dict:
    """
    Full analysis pipeline (OPTIMISED -- only 2 LLM API calls):
      Call 1: Combined summary + crime classification (multimodal with images)
      Call 2: Checklist analysis
    Plus rule-based cross-validation (no API).

    Returns dict with summary, classification, and checklist results.
    """
    call_delay = getattr(config, 'LLM_CALL_DELAY', 5)
    images = images or []

    # Preprocess text: OCR cleanup, whitespace normalization
    text = preprocess_text(text)

    # Smart truncation for very large documents
    processed_text = _truncate_text(text)

    logger.info(f"Processing document: {len(text):,} chars (truncated to {len(processed_text):,}), {len(images)} images")

    # -- API CALL 1: Combined Summary + Classification --
    # If images are present, we include them so Gemini can read scanned/handwritten pages
    if images:
        logger.info(f"API Call 1/2: Summary + Classification (multimodal: text + {len(images)} images)")
    else:
        logger.info("API Call 1/2: Summary + Classification (text only)")
    combined_prompt = _build_combined_prompt(processed_text)

    # Add image instruction to the prompt when images are present
    if images:
        image_instruction = (
            "\n\n## IMPORTANT: This document contains embedded images (scanned pages, handwritten documents, photos).\n"
            "The images are attached below. CAREFULLY read and extract ALL text from these images.\n"
            "These images may contain critical information like:\n"
            "- Victim/accused names and details\n"
            "- Witness statements\n"
            "- Investigation timeline and dates\n"
            "- Evidence descriptions\n"
            "- Medical/forensic reports\n"
            "- Case diary entries\n"
            "Include ALL information from both the text AND the images in your analysis.\n"
        )
        combined_prompt = combined_prompt + image_instruction

    combined_response = _call_llm(combined_prompt, SYSTEM_PROMPT, images=images)
    summary, llm_classification, ner_entities = _parse_combined_response(combined_response)

    # Heuristic field confidence (no API call)
    field_confidence = _compute_field_confidence(summary)

    # Rule-based cross-validation (FREE -- no API)
    rule_classification = detect_crime_type_rules(text)

    # Determine best crime type (4-category)
    if manual_crime_type:
        primary_crime = manual_crime_type
    else:
        primary_crime = llm_classification.get("primary_crime_type", "unknown")
        if rule_classification and primary_crime == "unknown":
            primary_crime = rule_classification[0]["crime_key"]
            llm_classification["primary_crime_type"] = primary_crime
        elif rule_classification and rule_classification[0]["crime_key"] != primary_crime:
            llm_classification["rule_based_suggestion"] = rule_classification[0]["crime_key"]

    # -- API CALL 2: Checklist Analysis --
    logger.info(f"API Call 2/2: Checklist for {primary_crime}")
    time.sleep(call_delay)

    checklist_result = {}
    crime_types_to_check = [primary_crime]
    secondary = llm_classification.get("secondary_crime_types", [])
    if secondary:
        crime_types_to_check.append(secondary[0])

    for ct in crime_types_to_check:
        if ct and ct != "unknown":
            checklist_result[ct] = analyse_checklist(processed_text, ct)
            break  # Only 1 LLM checklist call; secondary done via rules

    # Secondary types: rule-based checklist (no API)
    for ct in crime_types_to_check[1:]:
        if ct and ct != "unknown" and ct not in checklist_result:
            crime_info = get_crime_type_info(ct)
            if crime_info:
                checklist_result[ct] = _rule_based_checklist(
                    processed_text, crime_info["required_items"]
                )

    # -- RULE-BASED CROSS-CHECK: upgrade MISSING items if keywords found --
    for ct, cl_result in checklist_result.items():
        crime_info = get_crime_type_info(ct)
        if crime_info:
            rule_result = _rule_based_checklist(processed_text, crime_info["required_items"])
            checklist_result[ct] = _merge_checklist_with_rules(
                cl_result, rule_result
            )

    # -- STAGE 2B: Semantic Similarity Scoring (no API call) --
    similarity_results = {}
    for ct, cl_result in checklist_result.items():
        crime_info = get_crime_type_info(ct)
        if crime_info:
            logger.info(f"Computing semantic similarity for {ct}")
            sim_scores = compute_semantic_similarity(
                processed_text, crime_info["required_items"]
            )
            similarity_results[ct] = sim_scores
            # Merge similarity into checklist
            checklist_result[ct] = _merge_checklist_with_similarity(
                cl_result, sim_scores
            )

    # -- STAGE 3A: Confidence Scoring (no API call) --
    # Classification confidence
    classification_confidence = _compute_classification_confidence(
        llm_classification, rule_classification
    )
    llm_classification["composite_confidence"] = classification_confidence

    # Per-item checklist confidence
    for ct in checklist_result:
        checklist_result[ct] = _enrich_checklist_with_confidence(checklist_result[ct])

    return {
        "summary": summary,
        "classification": llm_classification,
        "rule_classification": rule_classification,
        "primary_crime_type": primary_crime,
        "checklists": checklist_result,
        "ner_entities": ner_entities,
        "similarity_scores": similarity_results,
        "field_confidence": field_confidence,
    }


def summarise_chargesheet(text: str) -> str:
    """
    Standalone summarisation (for notebook use).
    For the main pipeline, use process_chargesheet() which combines
    summary + classification in a single API call.
    """
    processed = _truncate_text(text)
    prompt = _build_summary_prompt(processed)
    return _call_llm(prompt, SYSTEM_PROMPT)


def _build_summary_prompt(text: str) -> str:
    """Standalone summary prompt (used by notebook)."""
    return f"""निम्नलिखित हिन्दी चार्जशीट/आरोप पत्र का विश्लेषण करें और संरचित सारांश दें।

## केस हेडर (Case Header)
- **FIR संख्या:**  **दिनांक:**  **थाना:**  **न्यायालय:**
- **घटना स्थल:**  **घटना दिनांक एवं समय:**

## पक्षकार (Parties Involved)
## घटना का सारांश (Incident Summary)
## लागू कानूनी धाराएं (Legal Sections Applied)
## प्रमुख साक्ष्य (Key Evidence)

---
{text}
"""


def classify_crime_type(text: str) -> dict:
    """Standalone classification (for notebook use)."""
    checklists = load_checklists()
    crime_types_str = "\n".join(
        f"- **{k}**: {v['display_name']} ({', '.join(v['typical_sections'])})"
        for k, v in checklists.items()
    )

    prompt = f"""चार्जशीट से अपराध का प्रकार वर्गीकृत करें।

श्रेणियाँ: {crime_types_str}

JSON में उत्तर दें:
```json
{{{{"primary_crime_type": "<key>", "secondary_crime_types": [], "detected_sections": [], "confidence": "high/medium/low", "reasoning": "..."}}}}
```

दस्तावेज़: {text[:5000]}
"""
    response = _call_llm(prompt, SYSTEM_PROMPT)
    json_match = re.search(r'\{[\s\S]*?\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {
        "primary_crime_type": "unknown", "secondary_crime_types": [],
        "detected_sections": [], "confidence": "low", "reasoning": response,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_checklist_output(checklist_result: dict, crime_key: str) -> str:
    """Format checklist analysis into a readable markdown string with semantic similarity scores."""
    crime_info = get_crime_type_info(crime_key)
    if not crime_info:
        return f"Unknown crime type: {crime_key}"

    items_by_id = {item["id"]: item for item in crime_info["required_items"]}
    checklist = checklist_result.get("checklist", [])

    present_items, missing_items, partial_items = [], [], []

    for entry in checklist:
        item_id = entry.get("id", "")
        status = entry.get("status", "missing")
        remarks = entry.get("remarks", "")
        page_no = entry.get("page_no", "")
        sim_score = entry.get("similarity_score", None)
        best_match = entry.get("best_match", "")
        item_confidence = entry.get("confidence", None)
        item_info = items_by_id.get(item_id, {"label_hi": item_id, "label_en": item_id})

        line = f"- {item_info['label_hi']} ({item_info['label_en']})"
        if page_no:
            line += f"  📄 **Pg {page_no}**"
        # Confidence badge
        if item_confidence is not None:
            if item_confidence >= 0.80:
                conf_badge = f"🟢 {item_confidence:.0%}"
            elif item_confidence >= 0.50:
                conf_badge = f"🟡 {item_confidence:.0%}"
            else:
                conf_badge = f"🔴 {item_confidence:.0%}"
            line += f"  [{conf_badge}]"
        if sim_score is not None:
            # Show similarity bar: ▓▓▓▓▒▒▒▒▒▒
            bar_len = 10
            filled = min(bar_len, max(0, round(sim_score * bar_len)))
            bar = "▓" * filled + "▒" * (bar_len - filled)
            line += f"  `{bar} {sim_score:.2f}`"
        if remarks:
            line += f" -- {remarks}"
        if best_match and status in ("present", "partial"):
            line += f"\n  > 📝 *\"{best_match}\"*"

        if status == "present":
            present_items.append(line)
        elif status == "partial":
            partial_items.append(line)
        else:
            missing_items.append(line)

    output = f"## चेकलिस्ट: {crime_info['display_name']}\n\n"

    output += f"### ✅ उपस्थित (Present) -- {len(present_items)} items\n"
    output += "\n".join(present_items) if present_items else "- कोई नहीं"
    output += "\n\n"

    if partial_items:
        output += f"### ⚠️ आंशिक (Partial) -- {len(partial_items)} items\n"
        output += "\n".join(partial_items)
        output += "\n\n"

    output += f"### ❌ अनुपस्थित / संभवतः गायब (Missing) -- {len(missing_items)} items\n"
    output += "\n".join(missing_items) if missing_items else "- कोई नहीं"
    output += "\n"

    return output


def format_classification_output(
    classification: dict,
    rule_results: list,
) -> str:
    """Format classification result into readable markdown."""
    output = ""

    # ── Crime Classification ──
    output += "## अपराध वर्गीकरण (Crime Classification)\n\n"

    checklists = load_checklists()
    primary = classification.get("primary_crime_type", "unknown")
    primary_info = checklists.get(primary, {})
    display = primary_info.get("display_name", primary)

    output += f"**प्राथमिक अपराध श्रेणी (Primary Crime Type):** {display}\n\n"

    # Composite confidence score with visual indicator
    composite_conf = classification.get("composite_confidence", None)
    if composite_conf is not None:
        bar_len = 20
        filled = min(bar_len, max(0, round(composite_conf * bar_len)))
        bar = "▓" * filled + "▒" * (bar_len - filled)
        if composite_conf >= 0.75:
            conf_label = "🟢 High"
        elif composite_conf >= 0.50:
            conf_label = "🟡 Medium"
        else:
            conf_label = "🔴 Low"
        output += f"**Classification Confidence:** {conf_label} `{bar}` **{composite_conf:.0%}**\n\n"
    else:
        confidence = classification.get("confidence", "")
        if confidence:
            emoji_map = {"high": "High", "medium": "Medium", "low": "Low"}
            output += f"**विश्वास स्तर (Confidence):** {emoji_map.get(confidence, confidence)}\n\n"

    secondary = classification.get("secondary_crime_types", [])
    if secondary:
        sec_names = ", ".join(
            checklists.get(s, {}).get("display_name", s) for s in secondary
        )
        output += f"**द्वितीयक श्रेणियाँ (Secondary):** {sec_names}\n\n"

    sections = classification.get("detected_sections", [])
    if sections:
        output += f"**पहचानी गई धाराएं (Detected Sections):** {', '.join(sections)}\n\n"

    reasoning = classification.get("reasoning", "")
    if reasoning:
        output += f"**कारण (Reasoning):** {reasoning}\n\n"

    if rule_results:
        output += "### नियम-आधारित विश्लेषण (Rule-based Analysis)\n"
        for r in rule_results[:5]:
            output += f"- {r['display_name']} (score: {r['score']})\n"
        output += "\n"

    return output


def format_ner_output(ner_entities) -> str:
    """
    Format NER entities into a compact, grid-style markdown.
    Uses inline tag badges instead of vertical bullet lists.
    """
    # Handle old dict format
    if isinstance(ner_entities, dict):
        if not ner_entities:
            return "⚠️ No entities extracted.\n"
        ner_entities = _convert_ner_to_flat_list(ner_entities)

    if not ner_entities:
        return "⚠️ No entities extracted.\n"

    # Group by type
    by_type = {}
    for ent in ner_entities:
        etype = ent.get("type", "UNKNOWN")
        by_type.setdefault(etype, []).append(ent["text"])

    # Display order and labels
    type_config = [
        ("ACCUSED",       "🔴 आरोपी",     "Accused"),
        ("WITNESS",       "🟡 गवाह",      "Witnesses"),
        ("OFFICER",       "👮 अधिकारी",   "Officers"),
        ("DOCTOR",        "🩺 चिकित्सक",  "Doctors"),
        ("PERSON",        "👤 अन्य व्यक्ति", "Other Persons"),
        ("DATE",          "📅 तिथि",      "Dates"),
        ("LOCATION",      "📍 स्थान",     "Locations"),
        ("LEGAL_SECTION", "⚖️ कानूनी धारा", "Legal Sections"),
        ("ORGANIZATION",  "🏛️ संगठन",    "Organizations"),
        ("LANDMARK",      "🏪 स्थल",      "Landmarks"),
        ("EVIDENCE",      "🔍 साक्ष्य",   "Evidence"),
        ("MONETARY",      "💰 धनराशि",    "Monetary"),
    ]

    total = len(ner_entities)
    type_count = len(by_type)

    output = f"## Named Entity Recognition (NER)\n"
    output += f"**{total} entities** across **{type_count} types**\n\n"

    # Summary bar: counts per type
    summary_parts = []
    for etype, hi_label, en_label in type_config:
        items = by_type.get(etype, [])
        if items:
            summary_parts.append(f"{hi_label} **{len(items)}**")
    if summary_parts:
        output += " · ".join(summary_parts) + "\n\n"
        output += "---\n\n"

    known_types = set()
    for etype, hi_label, en_label in type_config:
        known_types.add(etype)
        items = by_type.get(etype, [])
        if not items:
            continue

        output += f"**{hi_label} ({en_label}) — {len(items)}**\n\n"

        # Render as a compact table with 3 columns
        cols = 3
        output += "| | | | | | |\n"
        output += "|---|--------|---|--------|---|--------|\n"

        rows = (len(items) + cols - 1) // cols
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = r + c * rows
                if idx < len(items):
                    row_cells.append(f" {idx+1} | {items[idx]} ")
                else:
                    row_cells.append("  |  ")
            output += "|" + "|".join(row_cells) + "|\n"

        output += "\n"

    # Any extra types not in our config
    for etype, items in by_type.items():
        if etype not in known_types and items:
            output += f"**{etype} — {len(items)}**\n\n"
            output += "| | | | | | |\n"
            output += "|---|--------|---|--------|---|--------|\n"
            rows = (len(items) + 3 - 1) // 3
            for r in range(rows):
                row_cells = []
                for c in range(3):
                    idx = r + c * rows
                    if idx < len(items):
                        row_cells.append(f" {idx+1} | {items[idx]} ")
                    else:
                        row_cells.append("  |  ")
                output += "|" + "|".join(row_cells) + "|\n"
            output += "\n"

    return output


def format_timeline_output(ner_entities, summary: str = "") -> str:
    """
    Build a concise case timeline showing ONLY key case milestones.
    Filters aggressively: only dates with an identifiable case event
    (FIR, incident, arrest, medical, recovery, chargesheet, etc.)
    are included. Limits to ~15 most important entries.
    """
    if isinstance(ner_entities, dict):
        ner_entities = _convert_ner_to_flat_list(ner_entities) if ner_entities else []

    if not ner_entities:
        return "⚠️ No timeline data available.\n"

    import re as _re
    from datetime import datetime

    # Event patterns: (regex, emoji+label, priority 1=highest)
    _EVENT_PATTERNS = [
        (r'घटना|incident|वारदात|हमला|attack|मारपीट',               '⚡ घटना (Incident)', 1),
        (r'FIR|प्रथम\s*सूचना|एफ\.?आई\.?आर|मु\.?\s*अ\.?\s*नं|मुकदमा\s*दर्ज', '📋 FIR दर्ज (FIR Registered)', 2),
        (r'गिरफ्तार|arrest|धरपकड़|पकड़ा|गिरफतारी',              '🚔 गिरफ्तारी (Arrest)', 3),
        (r'पोस्टमार्टम|postmortem|शव\s*परीक्षण|autopsy',        '🏥 पोस्टमार्टम (Postmortem)', 3),
        (r'मेडिकल|medical|चिकित्सा|इलाज|परीक्षण\s*रिपोर्ट',    '🏥 मेडिकल जाँच (Medical)', 4),
        (r'बरामद|जब्त|seized|recovered|बरामदगी|कब्जा',          '📦 बरामदगी/जब्ती (Recovery)', 4),
        (r'चार्जशीट|chargesheet|आरोप\s*पत्र',                   '📑 चार्जशीट (Chargesheet)', 2),
        (r'जमानत|bail',                                          '⚖️ जमानत (Bail)', 5),
        (r'पुलिस\s*कस्टडी|remand|रिमांड|न्यायिक\s*हिरासत',      '⚖️ रिमांड/हिरासत (Remand)', 5),
        (r'पंचनामा|panchnama|पंचायतनामा|फर्द',                  '📋 पंचनामा (Panchnama)', 5),
        (r'मौका\s*मुआयना|spot|नक्शा|नक्सा|मौकानक्शा',          '🗺️ मौका मुआयना (Spot Visit)', 5),
        (r'बयान|कथन|statement|164',                              '📝 बयान (Statement)', 6),
        (r'FSL|फॉरेंसिक|forensic|विश्लेषण|DNA|CFSL',            '🔬 FSL/फॉरेंसिक (Forensic)', 5),
        (r'विवेचना|investigation|जाँच|जांच|तफ्तीश',             '🔍 विवेचना (Investigation)', 7),
        (r'पेशी|court|अदालत|न्यायालय|विचारण',                   '⚖️ पेशी (Court)', 6),
        (r'मृत्यु|death|मर\s*गय|शव',                            '💀 मृत्यु (Death)', 1),
        (r'रिपोर्ट|report',                                      '📄 रिपोर्ट (Report)', 7),
    ]

    date_pattern = re.compile(r'(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{2,4})')

    # Parse all valid dates
    all_dates = []
    seen = set()

    for ent in ner_entities:
        if ent.get("type") != "DATE":
            continue
        raw = ent["text"].strip()
        m = date_pattern.search(raw)
        if not m:
            continue

        day, month, year = m.group(1), m.group(2), m.group(3)
        if len(year) == 2:
            year = "20" + year
        try:
            dt_obj = datetime(int(year), int(month), int(day))
        except ValueError:
            continue

        date_key = dt_obj.strftime("%Y-%m-%d")
        if date_key in seen:
            continue
        seen.add(date_key)
        all_dates.append((raw, dt_obj))

    if not all_dates:
        return "⚠️ No valid dates found for timeline.\n"

    # For each date, find what case event it relates to
    def _classify_date(date_str: str, dt_obj: datetime, text: str):
        """Return (event_label, priority, context_snippet) or None if not important."""
        if not text:
            return None

        search_variants = [
            date_str,
            dt_obj.strftime("%d/%m/%Y"),
            dt_obj.strftime("%d.%m.%Y"),
            dt_obj.strftime("%d-%m-%Y"),
            dt_obj.strftime("%d/%m/%y"),
        ]

        for variant in search_variants:
            # Find ALL occurrences and pick the one with best event match
            start_pos = 0
            best_match = None

            while True:
                idx = text.find(variant, start_pos)
                if idx < 0:
                    break

                # Get surrounding context (120 chars each side)
                ctx_start = max(0, idx - 120)
                ctx_end = min(len(text), idx + len(variant) + 120)
                context = text[ctx_start:ctx_end]

                # Check against event patterns
                for pattern, label, priority in _EVENT_PATTERNS:
                    if _re.search(pattern, context, _re.IGNORECASE):
                        # Extract a clean 1-line description
                        before = text[max(0, idx - 80):idx]
                        after = text[idx + len(variant):min(len(text), idx + len(variant) + 120)]

                        # Find sentence boundary
                        line_start = before.rfind('\n')
                        line_start = max(line_start + 1, len(before) - 80) if line_start >= 0 else max(0, len(before) - 80)
                        line_end_candidates = [after.find('\n'), after.find('।')]
                        line_end_candidates = [x for x in line_end_candidates if x >= 0]
                        line_end = min(line_end_candidates) if line_end_candidates else min(100, len(after))

                        snippet = (before[line_start:].strip() + " " + after[:line_end].strip()).strip()
                        snippet = _re.sub(r'\s+', ' ', snippet).strip('━\n\t ।.-:, ')

                        # Truncate
                        if len(snippet) > 150:
                            snippet = snippet[:147] + "..."

                        if best_match is None or priority < best_match[1]:
                            best_match = (label, priority, snippet)
                        break

                start_pos = idx + 1

            if best_match:
                return best_match

        return None  # No meaningful event found — skip this date

    # Classify each date
    timeline_entries = []
    # Words that are NOT meaningful context (just the word "date" etc.)
    _NOISE_SNIPPETS = re.compile(
        r'^(दिनांक|तिथि|date|तारीख|दि\.|दिनाक)[\s.:;,।\-]*$',
        re.IGNORECASE
    )
    for raw, dt_obj in all_dates:
        result = _classify_date(raw, dt_obj, summary)
        if result:
            label, priority, snippet = result
            # Skip entries with empty or noise-only snippets
            if not snippet or _NOISE_SNIPPETS.match(snippet.strip()):
                continue
            timeline_entries.append((dt_obj, label, priority, snippet))

    if not timeline_entries:
        return "⚠️ No key case events identified for timeline.\n"

    # Sort by date
    timeline_entries.sort(key=lambda x: x[0])

    # If still too many (>15), keep only highest priority events
    if len(timeline_entries) > 15:
        timeline_entries.sort(key=lambda x: (x[2], x[0]))  # priority first, then date
        timeline_entries = timeline_entries[:15]
        timeline_entries.sort(key=lambda x: x[0])  # re-sort by date

    # Build visual HTML timeline
    # Color mapping for event types
    _EVENT_COLORS = {
        'Incident': '#e76f51',
        'FIR': '#0369a1',
        'Arrest': '#e63946',
        'Postmortem': '#457b9d',
        'Medical': '#457b9d',
        'Recovery': '#f4a261',
        'Chargesheet': '#0369a1',
        'Bail': '#6c757d',
        'Remand': '#6c757d',
        'Panchnama': '#264653',
        'Spot': '#264653',
        'Statement': '#7209b7',
        'Forensic': '#3a0ca3',
        'Investigation': '#264653',
        'Court': '#6c757d',
        'Death': '#e63946',
        'Report': '#264653',
    }

    def _get_event_color(label):
        for key, color in _EVENT_COLORS.items():
            if key.lower() in label.lower():
                return color
        return '#6c757d'

    # Build HTML
    output = '<div style="padding: 8px 0;">\n'

    # Date range header
    if len(timeline_entries) >= 2:
        earliest = timeline_entries[0][0]
        latest = timeline_entries[-1][0]
        span = (latest - earliest).days
        output += f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; padding:12px 16px; background:rgba(3,105,161,0.08); border-radius:10px; border:1px solid rgba(3,105,161,0.15);">\n'
        output += f'<span style="font-size:0.85rem; color:#6c757d;">From <strong style="color:#0369a1;">{earliest.strftime("%d %b %Y")}</strong></span>\n'
        output += f'<span style="font-size:0.8rem; color:#adb5bd;">&#9473;&#9473; {span} days &#9473;&#9473;</span>\n'
        output += f'<span style="font-size:0.85rem; color:#6c757d;">To <strong style="color:#0369a1;">{latest.strftime("%d %b %Y")}</strong></span>\n'
        output += '</div>\n'

    # Timeline flow
    output += '<div style="position:relative; padding-left:36px;">\n'
    # Vertical line
    output += '<div style="position:absolute; left:14px; top:6px; bottom:6px; width:2px; background:linear-gradient(180deg, #dee2e6 0%, #adb5bd 50%, #dee2e6 100%); border-radius:2px;"></div>\n'

    for i, (dt_obj, label, _prio, snippet) in enumerate(timeline_entries):
        color = _get_event_color(label)
        formatted_date = dt_obj.strftime("%d %b %Y")
        is_last = (i == len(timeline_entries) - 1)

        # Node dot
        output += f'<div style="position:relative; margin-bottom:{0 if is_last else 6}px; padding-bottom:{0 if is_last else 6}px;">\n'
        output += f'  <div style="position:absolute; left:-29px; top:8px; width:14px; height:14px; background:{color}; border-radius:50%; border:3px solid #fff; box-shadow:0 0 0 2px {color}40, 0 2px 8px rgba(0,0,0,0.1); z-index:1;"></div>\n'

        # Card
        output += f'  <div style="background:#fff; border:1px solid #e9ecef; border-left:3px solid {color}; border-radius:8px; padding:12px 16px; box-shadow:0 1px 4px rgba(0,0,0,0.04); transition:box-shadow 0.2s;">\n'
        output += f'    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">\n'
        output += f'      <span style="font-weight:700; font-size:0.82rem; color:{color}; text-transform:uppercase; letter-spacing:0.5px;">{label}</span>\n'
        output += f'      <span style="font-size:0.78rem; color:#adb5bd; font-weight:500;">{formatted_date}</span>\n'
        output += f'    </div>\n'
        if snippet:
            output += f'    <div style="font-size:0.88rem; color:#495057; line-height:1.5;">{snippet}</div>\n'
        output += f'  </div>\n'
        output += f'</div>\n'

    output += '</div>\n'
    output += '</div>\n'

    return output


def format_field_confidence(field_confidence: dict) -> str:
    """
    Format summary field confidence scores into a readable markdown table.
    Returns a section to append to the summary output.
    """
    if not field_confidence:
        return ""

    # Check if all values are -1 (no confidence data)
    if all(v == -1.0 for v in field_confidence.values()):
        return ""

    field_labels = {
        "fir_number": ("FIR संख्या", "FIR Number"),
        "fir_date": ("FIR दिनांक", "FIR Date"),
        "police_station": ("थाना", "Police Station"),
        "court": ("न्यायालय", "Court"),
        "place_of_occurrence": ("घटना स्थल", "Place of Occurrence"),
        "incident_date_time": ("घटना दिनांक/समय", "Incident Date/Time"),
        "complainant_name": ("शिकायतकर्ता", "Complainant Name"),
        "accused_names": ("आरोपी नाम", "Accused Names"),
        "witnesses": ("गवाह", "Witnesses"),
        "incident_summary": ("घटना सारांश", "Incident Summary"),
        "legal_sections": ("कानूनी धाराएं", "Legal Sections"),
        "key_evidence": ("प्रमुख साक्ष्य", "Key Evidence"),
    }

    output = "\n---\n\n## 📊 Summary Field Confidence Scores\n\n"
    output += "| Field | Confidence | Level |\n"
    output += "|-------|-----------|-------|\n"

    for field_key, (hi_label, en_label) in field_labels.items():
        score = field_confidence.get(field_key, -1.0)
        if score < 0:
            badge = "⚪ N/A"
            bar_str = "—"
        else:
            score = max(0.0, min(1.0, score))
            bar_len = 10
            filled = min(bar_len, max(0, round(score * bar_len)))
            bar_str = "▓" * filled + "▒" * (bar_len - filled) + f" {score:.0%}"
            if score >= 0.80:
                badge = "🟢 High"
            elif score >= 0.50:
                badge = "🟡 Medium"
            elif score >= 0.20:
                badge = "🟠 Low"
            else:
                badge = "🔴 Not found"
        output += f"| {hi_label} ({en_label}) | `{bar_str}` | {badge} |\n"

    # Overall average
    valid_scores = [v for v in field_confidence.values() if v >= 0]
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        output += f"\n**Overall Summary Confidence: {avg:.0%}**\n"

    return output
