# Smart Chargesheet Review & Summarisation Assistant

## 📌 Problem Statement

Indian police chargesheets (आरोप पत्र) are lengthy Hindi legal documents—often 50–200+ pages—containing FIR details, witness statements, evidence lists, medical reports, and case diary entries. Prosecutors, judges, and lawyers must manually review these to:
1. Understand the case summary
2. Identify the crime type and applicable legal sections
3. Verify that all required documents/evidence are attached

This is time-consuming, error-prone, and delays justice delivery.

**Our solution** automates all three tasks using an AI-powered multi-stage pipeline with engineered preprocessing, multimodal document understanding, and hybrid AI + rule-based validation.

## Quick Deploy

### Option 1: Render (recommended)
1. Push this repository to GitHub.
2. In Render, create a **Web Service** from the repository.
3. Render auto-detects `render.yaml` and uses Docker build from `Dockerfile`.
4. Set environment variables in Render:
    - `GEMINI_API_KEY` (required for Gemini)
    - `OPENAI_API_KEY` (optional, only if using OpenAI provider)
    - `LLM_PROVIDER` (`gemini` or `openai`)
5. Deploy. The app binds to `PORT` automatically.

### Option 2: Any Docker host (Railway/Fly/VM)
```bash
docker build -t chargesheet-assistant .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key -e LLM_PROVIDER=gemini chargesheet-assistant
```

App URL after start: `http://localhost:7860`

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Gradio)                   │
│   Upload .docx/.pdf/.txt  │  Paste text  │  Select crime type   │
│    API input     │  Auto-detect │  Regenerate checklist │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 1: DOCUMENT INGESTION & PREPROCESSING         │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Text Extract │  │ Page Markers │  │ Image Extraction        │ │
│  │ .docx → para │  │ [PAGE 1]     │  │ docx.part.rels → blobs │ │
│  │ .docx → tbl  │  │ [PAGE 2]     │  │ PDF page.images        │ │
│  │ .pdf → pages │  │ [PAGE N]     │  │ Filter <1KB icons      │ │
│  └─────────────┘  └──────────────┘  └──────────┬──────────────┘ │
│                                                  │               │
│                                      ┌───────────▼─────────────┐ │
│                                      │ Image Compression       │ │
│                                      │ Pillow resize ≤2048px   │ │
│                                      │ JPEG re-encode ≤1.5MB   │ │
│                                      │ RGBA→RGB conversion     │ │
│                                      │ Max 20 images           │ │
│                                      └─────────────────────────┘ │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 2: SMART TEXT PREPARATION                     │
│                                                                  │
│  Input: raw text (potentially millions of characters)            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ Boundary-Aware Truncation (if > 100K chars)              │    │
│  │                                                          │    │
│  │  ┌─────────────────┐        ┌──────────────────────┐     │    │
│  │  │ 60% from START  │  ...   │   40% from END       │     │    │
│  │  │ FIR header      │omitted │   Legal sections     │     │    │
│  │  │ Parties         │        │   Conclusions        │     │    │
│  │  │ Incident        │        │   Attachments list   │     │    │
│  │  └─────────────────┘        └──────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Why 60/40? Chargesheet headers contain FIR, parties, incident   │
│  details (high-value). Endings contain legal sections, evidence  │
│  lists, conclusions. Middle is often repetitive case diary.      │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 3: API CALL 1 — MULTIMODAL ANALYSIS           │
│                        (Summary + Classification)                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ PROMPT ENGINEERING                                         │  │
│  │                                                            │  │
│  │ System Prompt:                                             │  │
│  │  "You are a legal document analysis assistant              │  │
│  │   specialising in Indian police chargesheets..."           │  │
│  │                                                            │  │
│  │ Task 1: Structured Summary                                 │  │
│  │  → Case Header (FIR no, date, PS, court, location)        │  │
│  │  → Parties (complainant, accused, witnesses)               │  │
│  │  → Incident Summary (what/how/when/where)                  │  │
│  │  → Legal Sections (IPC/BNS/NDPS/Arms Act)                 │  │
│  │  → Key Evidence (seized items, reports)                    │  │
│  │                                                            │  │
│  │ Task 2: Crime Classification                               │  │
│  │  → 4 crime types with typical sections listed              │  │
│  │  → Output: JSON with primary_crime_type, secondary,        │  │
│  │    detected_sections, confidence, reasoning                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ MULTIMODAL INPUT CONSTRUCTION                              │  │
│  │                                                            │  │
│  │  contents = [                                              │  │
│  │    Part.from_text(system_prompt + task_prompt),            │  │
│  │    Part.from_bytes(image_1, "image/jpeg"),  ← scanned pg  │  │
│  │    Part.from_bytes(image_2, "image/png"),   ← case diary  │  │
│  │    ...                                                     │  │
│  │    Part.from_bytes(image_20, "image/jpeg"), ← med report  │  │
│  │  ]                                                         │  │
│  │                                                            │  │
│  │  MIME detection: magic bytes (89 PNG / FF D8 JPEG / etc)   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ RESILIENCE ENGINE                                          │  │
│  │                                                            │  │
│  │  Model Fallback Chain:                                     │  │
│  │   gemini-2.5-flash → 2.0-flash → 2.0-flash-lite          │  │
│  │   → 2.5-flash-lite → 1.5-flash                            │  │
│  │                                                            │  │
│  │  Multi-Key Rotation (round-robin):                         │  │
│  │   key1 → key2 → key3 → key1 (each has own quota)          │  │
│  │                                                            │  │
│  │  Strategy: For each model, try ALL keys before fallback    │  │
│  │                                                            │  │
│  │  Model Memory: _last_successful_model stored, used first   │  │
│  │                                                            │  │
│  │  429 Handling: exponential backoff (5s × 2^attempt)        │  │
│  │  404 Handling: skip model immediately                      │  │
│  │                                                            │  │
│  │  temperature=0 for deterministic output                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ RESPONSE PARSING                                           │  │
│  │                                                            │  │
│  │  1. Regex extract ```json ... ``` block                    │  │
│  │  2. Fallback: find {..."primary_crime_type"...} pattern    │  │
│  │  3. JSON cleanup: remove trailing commas                   │  │
│  │  4. Separate summary markdown from classification JSON     │  │
│  │  5. Strip Task 2 section from summary display              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│          STAGE 4: RULE-BASED CROSS-VALIDATION (NO API)           │
│                                                                  │
│  Runs entirely offline — zero API cost                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ For each of 4 crime types:                                 │  │
│  │                                                            │  │
│  │  1. Legal Section Matching (score +3 each)                 │  │
│  │     "धारा 392" / "Section 379" / "IPC 323" etc.           │  │
│  │                                                            │  │
│  │  2. Hindi Keyword Matching (score +2 each)                 │  │
│  │     "चोरी" / "मारपीट" / "गांजा" / "ऑनलाइन धोखाधड़ी"      │  │
│  │                                                            │  │
│  │  3. English Keyword Matching (score +1 each)               │  │
│  │     "robbery" / "assault" / "narcotic" / "cyber fraud"     │  │
│  │                                                            │  │
│  │  Result: Scored ranking of all 4 crime types               │  │
│  │  Used to validate/override LLM classification              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│          STAGE 5: API CALL 2 — CHECKLIST ANALYSIS                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Crime-Specific Checklist Loaded from checklists.json       │  │
│  │                                                            │  │
│  │  theft_robbery    → 20 required items                      │  │
│  │  assault_hurt     → 16 required items                      │  │
│  │  cyber_fraud      → 20 required items                      │  │
│  │  ndps             → 16 required items                      │  │
│  │                                                            │  │
│  │  Each item: id, label_hi, label_en                         │  │
│  │  e.g. {"id":"mlc", "label_hi":"MLC रिपोर्ट",              │  │
│  │        "label_en":"Medico Legal Certificate"}              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ PAGE-AWARE PROMPT                                          │  │
│  │                                                            │  │
│  │  "दस्तावेज़ में [PAGE N] मार्कर दिए गए हैं।               │  │
│  │   प्रत्येक present/partial वस्तु के लिए page_no           │  │
│  │   फ़ील्ड में बताएं कि यह किस पेज पर मिली"                │  │
│  │                                                            │  │
│  │  Output: JSON with status + page_no + remarks              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ FALLBACK: Rule-Based Keyword Checklist                     │  │
│  │                                                            │  │
│  │  If LLM JSON is unparseable or API fails:                  │  │
│  │  50+ Hindi/English keywords per checklist item             │  │
│  │                                                            │  │
│  │  "mlc" → ["MLC", "मेडिको लीगल", "medical"]               │  │
│  │  "weapon_desc" → ["हथियार", "चाकू", "लाठी", "weapon"]    │  │
│  │  "fsl_report" → ["FSL", "फॉरेंसिक", "forensic"]          │  │
│  │                                                            │  │
│  │  Total: ~50 keyword groups × 4-6 keywords each            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Secondary crime types get rule-based checklist (no extra API)   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              STAGE 6: OUTPUT FORMATTING & DISPLAY                │
│                                                                  │
│  Tab 1: Summary (Hindi Markdown)                                 │
│    केस हेडर → पक्षकार → घटना सारांश → धाराएं → साक्ष्य        │
│                                                                  │
│  Tab 2: Classification                                           │
│    Primary crime type + confidence + detected sections           │
│    Rule-based scores shown for transparency                      │
│                                                                  │
│  Tab 3: Checklist                                                │
│    ✅ Present (with 📄 Pg N)                                     │
│    ⚠️ Partial (with 📄 Pg N + what's missing)                    │
│    ❌ Missing                                                     │
│                                                                  │
│  Stats bar: character count, word count, page estimate,          │
│             image count, truncation warning                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Engineering Details — What Sets This Apart

### 1. Multimodal Document Understanding (not just text)

**Problem:** Hindi police chargesheets are often digitised by scanning handwritten case diary pages and pasting them as images inside a `.docx` file. A typical 18 MB `.docx` contains only ~19,000 characters of typed text (template fields) but **dozens of embedded images** with the actual case content — victim names, witness statements, evidence, investigation timeline.

**Our Solution:**
```
.docx file
  ├── Typed paragraph text (extracted via python-docx)
  ├── Table text (case diary tables — separately parsed)
  └── Embedded images (accessed via OPC relationship traversal)
       ├── doc.part.rels → filter "image" relationship type
       ├── target_part.blob → raw image bytes
       ├── Filter out icons < 1KB
       ├── Compress via Pillow (resize > 2048px, JPEG 80%)
       └── Send as Part.from_bytes() in Gemini multimodal call
```

**Why this matters:** Without image extraction, the system sees ~5% of the document's actual content. With it, accuracy jumped significantly because Gemini's vision reads handwritten Hindi text directly from scanned pages.

### 2. Page-Aware Processing

**Problem:** When a checklist says "MLC Report is present", the user needs to know *where* in a 100+ page document to find it.

**Our Solution:**
- **DOCX:** Parse Word's internal XML (`w:br type="page"` and `w:lastRenderedPageBreak`) to detect page boundaries and inject `[PAGE N]` markers into the extracted text
- **PDF:** Naturally page-based — inject `[PAGE N]` before each page's text
- **Checklist prompt** instructs the LLM to reference these markers in its `page_no` output field
- **Display:** Each checklist item shows 📄 **Pg N** for quick navigation

### 3. 2-Call Optimised Architecture

**Problem:** Naive approaches use 10-50+ API calls per document (one per chunk, one per task). This exceeds free-tier rate limits instantly.

**Our Solution:**

| Approach | API Calls | Why |
|----------|-----------|-----|
| Naive chunking | ~50+ | Split into small chunks, summarise each |
| Our approach | **2** | Gemini supports ~1M tokens natively |

- **Call 1:** Combined summary + crime classification (single prompt, two tasks)
- **Call 2:** Checklist analysis (after receiving crime type from Call 1)
- All rule-based validation is **free** (zero API calls)

### 4. API Resilience Engine

**Problem:** Free-tier Gemini has strict per-key rate limits (15 RPM / 1M TPM). With a single key and a single model, the system fails frequently.

**Our Solution — 3-layer resilience:**

```
Layer 1: Model Fallback Chain
  gemini-2.5-flash → 2.0-flash → 2.0-flash-lite → 2.5-flash-lite → 1.5-flash

Layer 2: Multi-Key Rotation (round-robin)
  key1 → key2 → key3 → key1 ...
  Each free-tier key has independent quota → N keys = N× capacity

Layer 3: Model Memory
  _last_successful_model = "gemini-2.5-flash"
  Next call starts with this model (no wasted retries)
```

**Strategy per call:**
```
For each model in chain:
  For each API key (round-robin):
    Try call (max 1 retry with 5s backoff)
    If 429 → next key
    If 404 → skip model entirely
    If success → remember model, advance key pointer, return
```

**Dynamic Model Discovery:** At startup, calls `client.models.list()` to discover which models are actually available for the user's API tier. Only tries models that exist.

### 5. Smart Text Truncation (Boundary-Aware)

**Problem:** Some documents exceed 1M characters. Naively cutting at a fixed point loses critical information.

**Our Solution:**
```
if len(text) > 100,000:
    keep = text[:60,000]           # 60% start — FIR header, parties, incident
         + "[...omitted...]"
         + text[-40,000:]          # 40% end — legal sections, conclusions
```

**Why 60/40?** Based on chargesheet structure analysis:
- **Start:** FIR number, date, police station, complainant/accused details, incident description (highest value)
- **Middle:** Repetitive case diary entries, investigation logs (lower value)
- **End:** Legal sections applied, conclusions, evidence list, attachments (high value)

### 6. Hybrid AI + Rule-Based Validation

**Problem:** LLMs can hallucinate or misclassify crime types. Pure keyword matching misses context.

**Our Solution — dual validation:**

| Component | Method | Cost | Purpose |
|-----------|--------|------|---------|
| LLM Classification | Gemini API | 1 call | Context-aware, handles ambiguity |
| Rule-Based Detection | Keyword + section matching | FREE | Cross-validation, fallback |
| LLM Checklist | Gemini API | 1 call | Accurate presence/absence with remarks |
| Rule-Based Checklist | ~250 Hindi/English keywords | FREE | Fallback if API fails |

**Cross-validation logic:**
```python
if llm_says == "unknown" and rules_say == "assault_hurt":
    use rules  # LLM failed, rules provide answer

if llm_says == "assault_hurt" and rules_say == "theft_robbery":
    show both  # Display rule-based suggestion for transparency
```

### 7. Deterministic Output

**Problem:** Same document producing different classifications on each run (e.g., "assault_hurt" one time, "theft_robbery" next time).

**Our Solution:** All Gemini API calls use `temperature=0`:
```python
config = types.GenerateContentConfig(temperature=0)
```
This ensures identical input always produces identical output — critical for legal applications where consistency matters.

### 8. Domain-Specific Checklist Engineering

Each crime type has a curated checklist of legally required documents, built from Indian criminal procedure knowledge:

| Crime Type | Items | Examples |
|------------|-------|---------|
| **Theft / Robbery** (चोरी/डकैती) | 20 | Property description, ownership proof, seizure memo, CCTV, site map |
| **Assault / Hurt** (मारपीट/चोट) | 16 | MLC report, injury certificate, weapon description, doctor's opinion |
| **Cyber Fraud** (साइबर धोखाधड़ी) | 20 | Transaction records, chat logs, device seizure, FSL cyber report |
| **NDPS** (नारकोटिक्स) | 16 | Substance description, NDPS §42/§50 compliance, weighment, FSL report |

Each item has:
- Bilingual labels (Hindi + English)
- Unique ID for JSON mapping
- Associated keywords for rule-based fallback (4-6 keywords per item across Hindi/English)

### 9. Robust Response Parsing

LLM output is unpredictable. Our parser handles:

```
1. Standard: ```json { ... } ```           → regex group(1)
2. No fences: {"primary_crime_type": ...}   → regex fallback
3. Trailing commas: {"a": 1,}               → regex cleanup
4. Mixed content: summary text + JSON       → split at Task 2 boundary
5. Complete failure                         → rule-based fallback (no crash)
```

### 10. MIME Type Detection via Magic Bytes

Images extracted from documents have no file extension. We detect format from the binary header:

```python
if img_bytes[:4] == b'\x89PNG':    → image/png
if img_bytes[:2] == b'\xff\xd8':   → image/jpeg
if img_bytes[:4] == b'GIF8':      → image/gif
if img_bytes[:2] == b'BM':        → image/bmp
```

This ensures correct `Content-Type` in the multimodal API request.

---

## 📊 Technology Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| **LLM** | Google Gemini 2.5 Flash | 1M token context, free tier, native vision, Hindi support |
| **Vision** | Gemini Multimodal (native) | No separate OCR library needed — reads Hindi handwriting directly |
| **UI** | Gradio 5.x | Rapid prototyping, file upload, tabbed output |
| **DOCX parsing** | python-docx + lxml | Paragraph, table, and OPC relationship access for images |
| **PDF parsing** | PyPDF2 | Text extraction + image extraction from PDF streams |
| **Image processing** | Pillow (PIL) | Compression, resize, RGBA→RGB, format conversion |
| **Page detection** | docx.oxml (w:br, w:lastRenderedPageBreak) | Accurate page boundary detection in Word XML |
| **Config** | Python module + env vars | Runtime-configurable API keys, models, thresholds |

---

## 🔁 Data Flow (End to End)

```
User uploads काण्ड_संख्या_0126.docx (18.1 MB)
    │
    ▼
extract_text_and_images()
    ├── paragraphs → 19,085 chars text (with [PAGE N] markers)
    ├── tables → additional case diary text
    └── images → 15 embedded scanned pages (compressed to ~8 MB total)
    │
    ▼
_truncate_text()
    └── 19K chars < 100K limit → no truncation needed
    │
    ▼
API CALL 1: process_chargesheet()
    ├── Prompt: system_prompt + combined_task_prompt + image_instruction
    ├── Contents: [text_part, image_1, image_2, ..., image_15]
    ├── Model: gemini-2.5-flash (remembered from last success)
    ├── Key: key2 (round-robin from key1 last time)
    ├── Config: temperature=0
    │
    ├── Response parsed → summary (markdown) + classification (JSON)
    └── _parse_combined_response() handles malformed JSON gracefully
    │
    ▼
detect_crime_type_rules() [FREE — no API]
    ├── Scans for "धारा 307", "IPC 323", "Arms Act" etc.
    ├── Scores: assault_hurt=16, theft_robbery=4, cyber=0, ndps=0
    └── Cross-validates with LLM result → match ✓
    │
    ▼
API CALL 2: analyse_checklist()
    ├── Loads assault_hurt checklist (16 items)
    ├── Prompt includes [PAGE N] markers for page referencing
    ├── Model: gemini-2.5-flash (same as Call 1 — remembered)
    ├── Key: key3 (advanced by round-robin)
    │
    ├── Response: JSON with status + page_no + remarks per item
    └── Fallback: 250+ keywords if JSON parsing fails
    │
    ▼
format_checklist_output()
    ├── ✅ Present (9): FIR details 📄 Pg 1, victim details 📄 Pg 3 ...
    ├── ⚠️ Partial (2): weapon description 📄 Pg 8 -- type unknown
    └── ❌ Missing (5): MLC report, injury certificate ...
    │
    ▼
Gradio UI renders 3 tabs: Summary | Classification | Checklist
```

---

## 📁 Project Structure

```
chargesheet-assistant/
├── processor.py          # Core engine (~1155 lines)
│   ├── Text + image extraction with page markers (docx/pdf)
│   ├── Image compression pipeline (Pillow)
│   ├── Smart boundary-aware truncation
│   ├── LLM interaction with 3-layer resilience engine
│   ├── Multimodal API call construction
│   ├── Prompt engineering (summary + classification + checklist)
│   ├── Response parsing with JSON cleanup
│   ├── Rule-based crime detection (250+ Hindi/English keywords)
│   └── Rule-based checklist fallback (50+ keyword groups)
│
├── app.py                # Gradio UI (~350 lines)
│   ├── Multi-key API input parsing
│   ├── File upload + text extraction orchestration
│   ├── Crime type selection (auto-detect / manual)
│   ├── Error handling with user-friendly rate-limit messages
│   └── Stats display (chars, words, pages, images)
│
├── config.py             # Configuration (~50 lines)
│   ├── LLM provider selection (Gemini / OpenAI)
│   ├── Multi-key support
│   ├── Processing thresholds (MAX_TEXT_LENGTH, delays)
│   └── Rate limit parameters
│
├── checklists.json       # Domain knowledge base
│   ├── 4 crime types × 16-20 items each
│   ├── Bilingual labels (Hindi + English)
│   ├── Typical legal sections per crime
│   └── Hindi/English keywords for rule-based detection
│
├── requirements.txt      # Dependencies
├── demo.ipynb            # Jupyter notebook demo
└── sample_*.txt          # Sample documents for each crime type
```

---

## ⚙️ Preprocessing Pipeline Summary

This is NOT a simple "input → API → output" system. Here's what happens **before** any API call:

| Step | What It Does | Why It Matters |
|------|-------------|----------------|
| 1. File parsing | python-docx / PyPDF2 reads document structure | Handles real-world .docx/.pdf formats |
| 2. Paragraph extraction | Iterates `doc.paragraphs` | Gets typed text content |
| 3. Table extraction | Iterates `doc.tables → rows → cells` | Case diary entries are often in tables |
| 4. Page break detection | Parses Word XML (`w:br`, `lastRenderedPageBreak`) | Enables page number references |
| 5. Page marker injection | Inserts `[PAGE N]` into text stream | LLM can reference specific pages |
| 6. Image relationship traversal | `doc.part.rels` → filter image relationships | Finds embedded scanned pages |
| 7. Icon filtering | Skip images < 1KB | Removes logos, bullets, decorators |
| 8. Image compression | Pillow resize + JPEG re-encode | Keeps API payload manageable |
| 9. RGBA→RGB conversion | PIL color mode conversion | JPEG doesn't support alpha channel |
| 10. Image limiting | Max 20 images | API payload size constraint |
| 11. MIME detection | Magic byte analysis (PNG/JPEG/GIF/BMP) | Correct Content-Type for API |
| 12. Smart truncation | 60/40 boundary-aware split | Preserves most valuable sections |
| 13. Keyword pre-scan | 250+ Hindi/English keyword matching | Rule-based backup classification |

**13 preprocessing steps** before the first API call is even made.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open http://localhost:7860
# Enter your Gemini API key(s) — comma-separated for multi-key rotation
# Upload a .docx or .pdf chargesheet
# Click "Analyse"
```

Get free API keys at: https://aistudio.google.com/apikey (3 keys recommended for 3× quota)

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| API calls per document | **2** (vs 50+ naive approach) |
| Processing time | ~30-60 seconds |
| Supported document size | Up to 200+ pages |
| Image handling | Up to 20 embedded images per document |
| Rate limit resilience | N API keys × 5 models = 5N attempts before failure |
| Offline capability | Rule-based classification + checklist works without API |
| Output consistency | Deterministic (temperature=0) |
| Preprocessing steps | 13 stages before first API call |
| Keyword database | 250+ Hindi/English legal keywords |
| Crime types supported | 4 (72 checklist items total) |

---

## ✅ Stage Completion Table

| Stage | Feature | Status | Details |
|-------|---------|--------|---------|
| **1A** | Structured Case Summary | ✅ Done | FIR header, parties, incident, legal sections, evidence |
| **1B** | Crime Classification | ✅ Done | 4 crime types + UNKNOWN fallback, hybrid LLM + 250-keyword rule engine |
| **1C** | Missing Items Checklist | ✅ Done | PRESENT/MISSING/PARTIAL with page numbers, LLM + rule-based fallback |
| **2A** | Named Entity Recognition | ✅ Done | 15 sub-types: PERSON_MAIN/RELATIVE/MENTIONED/OFFICER/DOCTOR, DATE_EVENT/PROCEDURE/HISTORY, LOCATION_EVENT/ADDRESS, LEGAL_SECTION, ORGANIZATION, LANDMARK, EVIDENCE, MONETARY. Post-processing validation: relationship phrase rejection, boundary cleanup, legal section hallucination filter |
| **2B** | Semantic Similarity Scoring | ✅ Done | TF-IDF (char 2-4 grams) + cosine similarity. Augments LLM checklist with confidence scores, best-matching sentences, and MISSING→PARTIAL upgrades for paraphrased mentions. Zero API cost. |
| — | Text Preprocessing | ✅ Done | OCR garbage removal, broken Devanagari cleanup, whitespace normalization, repeated header/footer removal, date format normalization |
| — | Multimodal Image Extraction | ✅ Done | OPC relationship traversal for .docx, Pillow compression, Gemini vision via Part.from_bytes |
| — | Page-Aware Processing | ✅ Done | Word XML parsing for page breaks, [PAGE N] markers, page numbers in checklist output |
| — | API Resilience Engine | ✅ Done | 3-layer: model fallback × key rotation × last-success memory. temperature=0 for deterministic output |
