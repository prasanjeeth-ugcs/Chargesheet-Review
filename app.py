"""
Smart Chargesheet Review & Summarisation Assistant – Gradio UI

Launch with:
    python app.py
"""

import os
import tempfile
import logging

import gradio as gr

import config

logger = logging.getLogger(__name__)
from processor import (
    extract_text,
    extract_text_and_images,
    process_chargesheet,
    format_checklist_output,
    format_classification_output,
    format_ner_output,
    format_field_confidence,
    format_timeline_output,
    list_crime_types,
    analyse_checklist,
    format_checklist_output,
    load_checklists,
    summarise_chargesheet,
    classify_crime_type,
    detect_crime_type_rules,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_uploaded_file(file) -> str:
    """Save an uploaded Gradio file obj and return its path."""
    if file is None:
        raise ValueError("No file uploaded.")
    # Gradio 4.x gives a filepath string
    if isinstance(file, str):
        return file
    return file.name


# ── Main Processing Function ─────────────────────────────────────────────────

def _parse_api_keys(api_key_str: str):
    """Parse comma-separated API keys and set them in config."""
    if not api_key_str or not api_key_str.strip():
        return
    keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    if config.LLM_PROVIDER.lower() == "gemini":
        config.GEMINI_API_KEY = keys[0]
        os.environ["GEMINI_API_KEY"] = keys[0]
        config.GEMINI_API_KEYS = keys
    else:
        config.OPENAI_API_KEY = keys[0]
        os.environ["OPENAI_API_KEY"] = keys[0]


def run_analysis(file, raw_text, manual_crime_type, api_key):
    """Main function called by the Gradio UI."""
    # Set API key(s) at runtime — supports comma-separated multiple keys
    _parse_api_keys(api_key)

    # Get text from file or raw text input
    text = ""
    images = []
    if file is not None:
        try:
            file_path = _save_uploaded_file(file)
            text, images = extract_text_and_images(file_path)
            if images:
                logger.info(f"Extracted {len(images)} images from document")
        except Exception as e:
            return (
                f"❌ Error reading file: {e}",
                "",
                "",
                "",
                "",
            )
    elif raw_text and raw_text.strip():
        text = raw_text.strip()
    else:
        return (
            "❌ Please upload a file or paste text.",
            "",
            "",
            "",
            "",
            "",
        )

    if len(text) < 100:
        return (
            "❌ Document too short. Please provide a complete chargesheet.",
            "",
            "",
            "",
            "",
            "",
        )

    # Resolve manual crime type
    override_crime = None
    if manual_crime_type and manual_crime_type != "Auto-detect":
        # Extract key from display string
        crime_types = list_crime_types()
        for ct in crime_types:
            if ct["display_name"] in manual_crime_type or ct["display_name_en"] in manual_crime_type:
                override_crime = ct["key"]
                break

    try:
        results = process_chargesheet(text, manual_crime_type=override_crime, images=images)
    except Exception as e:
        error_str = str(e)
        is_rate_limit = any(kw in error_str for kw in [
            "429", "RESOURCE_EXHAUSTED", "rate_limit", "quota", "Too Many Requests",
        ])
        if is_rate_limit:
            error_msg = (
                "❌ **Rate Limit Exceeded**\n\n"
                "All API keys have exhausted their quota.\n\n"
                "**Solutions:**\n"
                "1. **Add more API keys** — comma-separated (e.g. key1, key2, key3)\n"
                "   - Create free keys at [Google AI Studio](https://aistudio.google.com/apikey)\n"
                "   - Each key has its own quota — 3 keys = 3x capacity!\n"
                "2. **Wait 1-2 minutes** and try again\n\n"
                f"Technical details: `{error_str[:300]}`"
            )
        else:
            error_msg = f"❌ Analysis error: {error_str[:500]}"
        return (
            error_msg,
            "",
            "",
            "",
            "",
            f"⚠️ Document size: {len(text):,} characters ({len(text.split()):,} words)",
        )

    # Format outputs
    summary = results["summary"]

    # Append field confidence table to summary
    field_conf = results.get("field_confidence", {})
    if field_conf:
        summary += format_field_confidence(field_conf)

    classification_md = format_classification_output(
        results["classification"],
        results["rule_classification"],
    )

    checklist_md = ""
    for crime_key, cl_result in results["checklists"].items():
        checklist_md += format_checklist_output(cl_result, crime_key)
        checklist_md += "\n---\n\n"

    # NER output
    ner_md = format_ner_output(results.get("ner_entities", []))

    # Timeline output — use full document text for richer context
    timeline_md = format_timeline_output(
        results.get("ner_entities", []),
        text,
    )

    # Stats
    max_len = getattr(config, 'MAX_TEXT_LENGTH', 80000)
    was_truncated = len(text) > max_len
    stats = (
        f"📊 **Document Stats**\n"
        f"- Characters: {len(text):,}\n"
        f"- Est. words: {len(text.split()):,}\n"
        f"- Est. pages: {max(1, len(text) // 2500)}\n"
        f"- Embedded images: {len(images)}\n"
    )
    if was_truncated:
        stats += f"- ⚠️ Truncated to {max_len:,} chars for processing\n"

    return summary, classification_md, checklist_md, ner_md, timeline_md, stats


def generate_report(file, raw_text, api_key_str, summary_md, classification_md, checklist_md, ner_md, timeline_md):
    """Generate a downloadable combined report from all analysis outputs."""
    if not summary_md or summary_md.startswith("\u274c"):
        return gr.update(value=None, visible=False)

    report = "# Chargesheet Analysis Report\n---\n\n"

    # Add document info
    if file is not None:
        doc_name = os.path.basename(file) if isinstance(file, str) else "Uploaded Document"
        report += f"**Document:** {doc_name}\n\n"

    from datetime import datetime
    report += f"**Report Generated:** {datetime.now().strftime('%d %b %Y, %H:%M')}\n\n"
    report += "---\n\n"

    sections = [
        ("1. Summary", summary_md),
        ("2. Crime Classification", classification_md),
        ("3. Missing Documents Checklist", checklist_md),
        ("4. Named Entity Recognition", ner_md),
        ("5. Case Timeline", timeline_md),
    ]
    for title, content in sections:
        report += f"# {title}\n\n{content}\n\n---\n\n"

    report += "> *This report was auto-generated by the Smart Chargesheet Review & Summarisation Assistant.*\n"

    # Save to temp file
    report_path = os.path.join(tempfile.gettempdir(), "chargesheet_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return gr.update(value=report_path, visible=True)


def regenerate_checklist(file, raw_text, selected_crime_type, api_key):
    """Regenerate checklist with a manually selected crime type."""
    _parse_api_keys(api_key)

    text = ""
    if file is not None:
        try:
            file_path = _save_uploaded_file(file)
            text = extract_text(file_path)
        except Exception as e:
            return f"❌ Error: {e}"
    elif raw_text and raw_text.strip():
        text = raw_text.strip()
    else:
        return "❌ No document provided."

    if not selected_crime_type or selected_crime_type == "Auto-detect":
        return "❌ Please select a crime type."

    crime_types = list_crime_types()
    crime_key = None
    for ct in crime_types:
        if ct["display_name"] in selected_crime_type or ct["display_name_en"] in selected_crime_type:
            crime_key = ct["key"]
            break

    if not crime_key:
        return f"❌ Unknown crime type: {selected_crime_type}"

    try:
        result = analyse_checklist(text, crime_key)
        return format_checklist_output(result, crime_key)
    except Exception as e:
        return f"❌ Error: {e}"


# ── Build Gradio Interface ───────────────────────────────────────────────────

def build_app():
    crime_types = list_crime_types()
    crime_type_choices = ["Auto-detect"] + [
        f"{ct['display_name_en']}" for ct in crime_types
    ]

    with gr.Blocks(
        title=config.APP_TITLE,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.sky,
            secondary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.stone,
            font=("Inter", "system-ui", "sans-serif"),
            font_mono=("JetBrains Mono", "Fira Code", "monospace"),
        ).set(
            body_background_fill="#f5f5f0",
            body_background_fill_dark="#f5f5f0",
            body_text_color="#2b2d42",
            body_text_color_dark="#2b2d42",
            block_background_fill="#ffffff",
            block_background_fill_dark="#ffffff",
            block_border_width="1px",
            block_border_color="#e5e7eb",
            block_border_color_dark="#e5e7eb",
            block_label_text_color="#6c757d",
            block_label_text_color_dark="#6c757d",
            block_title_text_color="#2b2d42",
            block_title_text_color_dark="#2b2d42",
            input_background_fill="#ffffff",
            input_background_fill_dark="#ffffff",
            input_border_color="#d1d5db",
            input_border_color_dark="#d1d5db",
            input_placeholder_color="#9ca3af",
            button_primary_background_fill="linear-gradient(135deg, #0369a1 0%, #0284c7 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, #0369a1 0%, #0284c7 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%)",
            button_primary_text_color="#ffffff",
            button_primary_border_color="transparent",
            button_secondary_background_fill="#ffffff",
            button_secondary_background_fill_dark="#ffffff",
            button_secondary_background_fill_hover="#f0f9ff",
            button_secondary_background_fill_hover_dark="#f0f9ff",
            button_secondary_text_color="#0369a1",
            button_secondary_border_color="#d1d5db",
            border_color_primary="#0369a1",
            shadow_drop="0 1px 3px rgba(0,0,0,0.08)",
            shadow_drop_lg="0 4px 14px rgba(0,0,0,0.08)",
            checkbox_background_color="#ffffff",
            checkbox_background_color_dark="#ffffff",
            color_accent="#0ea5e9",
            color_accent_soft="#e0f2fe",
            link_text_color="#0284c7",
            link_text_color_hover="#0ea5e9",
            link_text_color_dark="#0284c7",
            link_text_color_hover_dark="#0ea5e9",
            link_text_color_visited="#0369a1",
            link_text_color_visited_dark="#0369a1",
            link_text_color_active="#38bdf8",
            link_text_color_active_dark="#38bdf8",
        ),
        css="""
        /* ══════════════════════════════════════════════════════
           CLEAN LIGHT PALETTE — SKY BLUE
           Sky Blue:  #0369a1 / #0284c7 / #0ea5e9 / #7dd3fc / #e0f2fe
           Orange:    #e76f51 / #f4845f / #f4a261
           BG:        #f5f5f0 / #ffffff / #fafaf8
           Text:      #2b2d42 / #495057 / #6c757d
           Border:    #e5e7eb / #d1d5db
        ══════════════════════════════════════════════════════ */

        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

        /* ── Global ── */
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
            background: #f5f5f0 !important;
        }

        .gradio-container p, .gradio-container li, .gradio-container td,
        .gradio-container input, .gradio-container textarea, .gradio-container select,
        .gradio-container button, .gradio-container label, .gradio-container span {
            font-family: 'Inter', system-ui, sans-serif !important;
        }
        .gradio-container h1, .gradio-container h2, .gradio-container h3 {
            font-family: 'DM Sans', 'Inter', sans-serif !important;
        }

        /* ── Hero Header ── */
        .hero-header {
            background: linear-gradient(135deg, #7dd3fc 0%, #bae6fd 40%, #e0f2fe 100%);
            border: none;
            border-radius: 20px;
            padding: 40px 32px 32px;
            margin-bottom: 28px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(125, 211, 252, 0.2);
        }
        .hero-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 50%, rgba(255,255,255,0.05) 0%, transparent 50%);
            pointer-events: none;
        }
        .hero-header::after {
            display: none;
        }
        .hero-header h1 {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #0c4a6e !important;
            -webkit-text-fill-color: #0c4a6e !important;
            background: none !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.3px;
        }
        .hero-header p {
            color: #1e3a5f !important;
            font-size: 0.95rem !important;
            margin: 4px 0 !important;
        }
        .hero-header .tagline {
            color: #0369a1 !important;
            -webkit-text-fill-color: #0369a1 !important;
            background: none !important;
            font-weight: 700;
            font-size: 1.05rem !important;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
        .hero-header .divider {
            width: 120px;
            height: 2px;
            background: linear-gradient(90deg, transparent, #0284c7, transparent);
            margin: 12px auto;
        }

        /* ── Section Headers ── */
        .section-label h3 {
            color: #0369a1 !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            border-bottom: 2px solid;
            border-image: linear-gradient(90deg, #0369a1, #7dd3fc, transparent) 1;
            padding-bottom: 10px;
            margin-bottom: 14px !important;
        }

        /* ── Input Panel ── */
        .input-panel {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 14px !important;
            padding: 10px !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .input-panel .gr-input, .input-panel textarea, .input-panel input {
            background: #fafaf8 !important;
            border-color: #d1d5db !important;
            color: #2b2d42 !important;
            border-radius: 8px !important;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        .input-panel .gr-input:focus, .input-panel textarea:focus, .input-panel input:focus {
            border-color: #0284c7 !important;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.12), 0 0 12px rgba(2, 132, 199, 0.06) !important;
        }

        /* ── Buttons ── */
        .action-buttons button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px;
            transition: all 0.25s ease;
            padding: 11px 26px !important;
            font-size: 0.85rem !important;
        }
        .action-buttons button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(3, 105, 161, 0.2);
        }
        .action-buttons .primary {
            min-width: 140px;
        }

        /* ── Results Panel ── */
        .results-panel {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 14px !important;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }

        /* ── Tabs ── */
        .gr-tab-nav {
            background: #fafaf8 !important;
            border-bottom: 1px solid #e5e7eb !important;
            border-radius: 14px 14px 0 0 !important;
            padding: 6px 10px 0 !important;
            gap: 2px !important;
        }
        .gr-tab-nav button {
            background: transparent !important;
            color: #9ca3af !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 16px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.3px;
            transition: all 0.25s ease !important;
        }
        .gr-tab-nav button:hover {
            color: #0284c7 !important;
            background: #f0f9ff !important;
        }
        .gr-tab-nav button.selected {
            color: #0369a1 !important;
            border-bottom-color: transparent !important;
            background: #ffffff !important;
            box-shadow: inset 0 -2px 0 0 #0369a1;
        }

        /* ── Output Box ── */
        .output-box {
            min-height: 300px;
            padding: 26px !important;
            background: #ffffff !important;
            border-radius: 0 0 12px 12px !important;
            border: 1px solid #e5e7eb !important;
            border-top: none !important;
            line-height: 1.8;
            color: #2b2d42 !important;
        }
        .output-box h2 {
            color: #0369a1 !important;
            -webkit-text-fill-color: #0369a1 !important;
            background: none !important;
            font-weight: 700 !important;
            font-size: 1.35rem !important;
            margin-top: 22px !important;
            margin-bottom: 14px !important;
            padding-bottom: 8px;
            border-bottom: 2px solid #e0f2fe;
            letter-spacing: 0.3px;
        }
        .output-box h3 {
            color: #0284c7 !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            margin-top: 16px !important;
            letter-spacing: 0.3px;
        }
        .output-box strong {
            color: #1a1a2e !important;
        }
        .output-box blockquote {
            border-left: 3px solid #0ea5e9 !important;
            background: linear-gradient(90deg, rgba(224,242,254,0.4) 0%, rgba(255,255,255,0) 100%) !important;
            padding: 12px 18px !important;
            margin: 10px 0 !important;
            border-radius: 0 10px 10px 0 !important;
            color: #495057 !important;
        }
        .output-box table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 10px;
            overflow: hidden;
            margin: 14px 0;
            border: 1px solid #e5e7eb;
        }
        .output-box table th {
            background: #f0f9ff !important;
            color: #0369a1 !important;
            font-weight: 600;
            padding: 12px 14px;
            text-align: left;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'Inter', sans-serif !important;
        }
        .output-box table td {
            padding: 10px 14px;
            border-top: 1px solid #f3f4f6;
            color: #495057;
            font-size: 0.9rem;
        }
        .output-box table tr:nth-child(even) td {
            background: #fafaf8;
        }
        .output-box table tr:hover td {
            background: #f0f9ff;
        }

        /* ── Stats Card ── */
        .stats-card {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            padding: 18px !important;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .stats-card::before {
            content: '';
            position: absolute;
            top: -10px; right: -10px;
            width: 100px; height: 100px;
            background: radial-gradient(circle, rgba(3,105,161,0.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .stats-card p, .stats-card li {
            color: #6c757d !important;
            font-size: 0.9rem !important;
        }
        .stats-card strong {
            color: #0369a1 !important;
        }

        /* ── Download Section ── */
        .download-section {
            margin-top: 16px;
        }
        .download-section button {
            background: linear-gradient(135deg, #e76f51 0%, #f4845f 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 13px 32px !important;
            transition: all 0.25s ease !important;
            width: 100%;
            letter-spacing: 0.5px;
        }
        .download-section button:hover {
            background: linear-gradient(135deg, #f4845f 0%, #f4a261 100%) !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(231, 111, 81, 0.3) !important;
        }

        /* ── Report File ── */
        .report-file {
            margin-top: 8px;
        }
        .report-file .gr-file {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 10px !important;
        }

        /* ── Footer ── */
        .footer-section {
            background: #ffffff !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 14px !important;
            padding: 24px 28px !important;
            margin-top: 12px !important;
            position: relative;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .footer-section::before {
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #0369a1, #0ea5e9, #0284c7, transparent);
            opacity: 0.6;
        }
        .footer-section h3 {
            color: #0369a1 !important;
            -webkit-text-fill-color: #0369a1 !important;
            background: none !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            margin-bottom: 10px !important;
            letter-spacing: 0.5px;
        }
        .footer-section li, .footer-section p {
            color: #6c757d !important;
            font-size: 0.85rem !important;
            line-height: 1.7;
        }
        .footer-section strong {
            color: #495057 !important;
        }
        .footer-section a {
            color: #0369a1 !important;
            text-decoration: none;
            border-bottom: 1px dotted rgba(3, 105, 161, 0.4);
            transition: all 0.2s;
        }
        .footer-section a:hover {
            color: #0284c7 !important;
            border-bottom-color: #0284c7;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #f5f5f0; }
        ::-webkit-scrollbar-thumb {
            background: #d1d5db;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

        /* ── Code blocks ── */
        .output-box code {
            background: #f0f9ff !important;
            color: #0369a1 !important;
            padding: 2px 7px;
            border-radius: 5px;
            font-size: 0.85em;
            border: 1px solid #e0f2fe;
        }

        /* ── Markdown lists ── */
        .output-box ul, .output-box ol {
            padding-left: 20px;
        }
        .output-box li {
            margin-bottom: 5px;
            color: #495057;
        }
        .output-box li::marker {
            color: #0284c7;
        }

        /* ── HR / Dividers ── */
        .output-box hr {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #d1d5db, #7dd3fc, #d1d5db, transparent);
            margin: 22px 0;
        }

        /* ── File component ── */
        .gr-file { border-radius: 10px !important; }

        /* ── Responsive ── */
        @media (max-width: 768px) {
            .hero-header h1 { font-size: 1.5rem !important; }
            .gr-tab-nav button { padding: 8px 10px !important; font-size: 0.78rem !important; }
        }
        """,
    ) as app:
        # ── Hero Header ──
        gr.Markdown(
            f"""
            <div class="hero-header">
                <h1>Smart Chargesheet Review & Summarisation Assistant</h1>
                <div class="divider"></div>
                <p class="tagline">AI-Powered Legal Document Analysis</p>
                <p>Upload a Hindi chargesheet &rarr; Get structured summary, crime classification, NER & missing documents checklist</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # ── Left: Input Panel ──
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Input", elem_classes=["section-label"])

                with gr.Group(elem_classes=["input-panel"]):
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your Gemini API key",
                        type="password",
                        info=f"Provider: {config.LLM_PROVIDER.upper()}",
                    )

                    file_input = gr.File(
                        label="Upload Chargesheet",
                        file_types=[".txt", ".pdf", ".docx"],
                        type="filepath",
                    )

                    text_input = gr.Textbox(
                        label="Or Paste Text",
                        placeholder="Paste chargesheet text here...",
                        lines=8,
                        max_lines=20,
                    )

                    crime_type_dropdown = gr.Dropdown(
                        choices=crime_type_choices,
                        value="Auto-detect",
                        label="Crime Type",
                        info="Auto-detect or manually override",
                    )

                with gr.Row(elem_classes=["action-buttons"]):
                    analyse_btn = gr.Button(
                        "Analyse",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary"],
                    )
                    regen_btn = gr.Button(
                        "Regenerate Checklist",
                        variant="secondary",
                    )

                stats_output = gr.Markdown(
                    label="Stats",
                    elem_classes=["stats-card"],
                )

            # ── Right: Results Panel ──
            with gr.Column(scale=2, min_width=600):
                gr.Markdown("### Analysis Results", elem_classes=["section-label"])

                with gr.Group(elem_classes=["results-panel"]):
                    with gr.Tabs():
                        with gr.TabItem("Summary"):
                            summary_output = gr.Markdown(
                                label="Summary",
                                elem_classes=["output-box"],
                            )

                        with gr.TabItem("Classification"):
                            classification_output = gr.Markdown(
                                label="Classification",
                                elem_classes=["output-box"],
                            )

                        with gr.TabItem("Checklist"):
                            checklist_output = gr.Markdown(
                                label="Checklist",
                                elem_classes=["output-box"],
                            )

                        with gr.TabItem("NER Entities"):
                            ner_output = gr.Markdown(
                                label="NER",
                                elem_classes=["output-box"],
                            )

                        with gr.TabItem("Case Timeline"):
                            timeline_output = gr.Markdown(
                                label="Timeline",
                                elem_classes=["output-box"],
                            )

                # Download button + hidden file that appears after generation
                with gr.Row(elem_classes=["download-section"]):
                    download_btn = gr.Button(
                        "Download Full Report",
                        variant="secondary",
                        size="lg",
                    )
                with gr.Column(elem_classes=["report-file"]):
                    report_file = gr.File(
                        visible=False,
                        interactive=False,
                        label="Report Ready",
                    )

        # Wire up events
        analyse_btn.click(
            fn=run_analysis,
            inputs=[file_input, text_input, crime_type_dropdown, api_key_input],
            outputs=[summary_output, classification_output, checklist_output, ner_output, timeline_output, stats_output],
        )

        regen_btn.click(
            fn=regenerate_checklist,
            inputs=[file_input, text_input, crime_type_dropdown, api_key_input],
            outputs=[checklist_output],
        )

        download_btn.click(
            fn=generate_report,
            inputs=[file_input, text_input, api_key_input, summary_output, classification_output, checklist_output, ner_output, timeline_output],
            outputs=[report_file],
        )

        # Footer
        gr.Markdown(
            """
            <div class="footer-section">
            <h3>How to Use</h3>
            <ol>
            <li>Enter your <strong>Gemini API Key</strong> &mdash; get one free at <a href="https://aistudio.google.com/apikey" target="_blank">Google AI Studio</a></li>
            <li><strong>Upload</strong> a chargesheet (.txt / .pdf / .docx) or <strong>paste text</strong></li>
            <li>Click <strong>Analyse</strong> &mdash; results appear across five tabs</li>
            <li>Click <strong>Download Full Report</strong> to save a combined markdown file</li>
            <li>(Optional) Override crime type and click <strong>Regenerate Checklist</strong></li>
            </ol>
            <h3>Technical Highlights</h3>
            <p>Multi-Key Rotation &bull; Gemini 2.5 Flash with 1M context &bull; Hybrid LLM + Rule Classification &bull; Semantic Similarity Matching &bull; Model Fallback Chain &bull; Hindi Output</p>
            </div>
            """
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
