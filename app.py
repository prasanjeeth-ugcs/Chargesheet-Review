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
    summary = results.get("summary", "⚠️ No summary generated.")

    # Append field confidence table to summary
    try:
        field_conf = results.get("field_confidence", {})
        if field_conf:
            summary += format_field_confidence(field_conf)

        confidence_scores = results.get("confidence_scores", {})
        if confidence_scores:
            ocr = confidence_scores.get("ocr_confidence", 0.0)
            ext = confidence_scores.get("extraction_confidence", 0.0)
            cls = confidence_scores.get("classification_confidence", 0.0)
            ocr_warn_threshold = float(getattr(config, "OCR_LOW_QUALITY_THRESHOLD", 0.70))
            if ocr < ocr_warn_threshold:
                summary += (
                    "\n\n> ⚠️ **OCR Quality Warning:** Document scan quality appears low. "
                    "Some names/sections/dates may need manual review."
                )
            summary += (
                "\n---\n\n"
                "## 🎯 Pipeline Confidence Scores\n\n"
                f"- OCR Confidence: **{ocr:.0%}**\n"
                f"- Extraction Confidence: **{ext:.0%}**\n"
                f"- Classification Confidence: **{cls:.0%}**\n"
            )
    except Exception as e:
        logger.error(f"Field confidence formatting error: {e}", exc_info=True)

    try:
        classification_md = format_classification_output(
            results["classification"],
            results["rule_classification"],
        )
    except Exception as e:
        logger.error(f"Classification formatting error: {e}", exc_info=True)
        classification_md = f"⚠️ Error formatting classification: {e}"

    checklist_md = ""
    try:
        for crime_key, cl_result in results["checklists"].items():
            checklist_md += format_checklist_output(cl_result, crime_key)
            checklist_md += "\n---\n\n"
    except Exception as e:
        logger.error(f"Checklist formatting error: {e}", exc_info=True)
        checklist_md = f"⚠️ Error formatting checklist: {e}"

    # NER output
    try:
        ner_md = format_ner_output(results.get("ner_entities", []))
    except Exception as e:
        logger.error(f"NER formatting error: {e}", exc_info=True)
        ner_md = f"⚠️ Error formatting NER output: {e}"

    # Timeline output — use OCR-cleaned preprocessed text for richer context
    # (raw text may still contain ि/न corruption which leaks into snippets)
    try:
        clean_text = results.get("preprocessed_text", text)
        timeline_md = format_timeline_output(
            results.get("ner_entities", []),
            summary,
            clean_text,
        )
    except Exception as e:
        logger.error(f"Timeline formatting error: {e}", exc_info=True)
        timeline_md = f"⚠️ Error formatting timeline: {e}"

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

    # Log output sizes for debugging
    logger.info(f"UI outputs: summary={len(summary)} chars, classification={len(classification_md)} chars, "
                f"checklist={len(checklist_md)} chars, ner={len(ner_md)} chars, "
                f"timeline={len(timeline_md)} chars")

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
        theme=gr.themes.Soft(),
        css=".sticky-panel { position: sticky; top: 20px; align-self: flex-start; }",
    ) as app:
        # ── Header ──
        gr.Markdown(
            """
            <div style="text-align: center; padding: 20px 0 10px;">
                <h1 style="margin-bottom: 5px;">Smart Chargesheet Review & Summarisation Assistant</h1>
                <p style="font-size: 1.1rem; opacity: 0.8;"><strong>AI-Powered Legal Document Analysis</strong></p>
                <p style="font-size: 0.95rem; opacity: 0.7;">Upload a Hindi chargesheet → Get structured summary, crime classification, NER & missing documents checklist</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # ── Left: Input Panel ──
            with gr.Column(scale=1, min_width=350, elem_classes=["sticky-panel"]):
                gr.Markdown("### 📁 Input")

                with gr.Group():
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
                        lines=6,
                        max_lines=12,
                    )

                    crime_type_dropdown = gr.Dropdown(
                        choices=crime_type_choices,
                        value="Auto-detect",
                        label="Crime Type",
                        info="Auto-detect or manually override",
                    )

                with gr.Row():
                    analyse_btn = gr.Button(
                        "🔍 Analyse",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    regen_btn = gr.Button(
                        "🔄 Regenerate",
                        variant="secondary",
                        scale=1,
                    )

                stats_output = gr.Markdown(label="Processing Stats")

            # ── Right: Results Panel ──
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 📊 Analysis Results")

                # Processing status indicator
                processing_status = gr.HTML(visible=False)

                with gr.Tabs():
                    with gr.TabItem("📝 Summary"):
                        summary_output = gr.Markdown(label="Summary")

                    with gr.TabItem("🏷️ Classification"):
                        classification_output = gr.Markdown(label="Classification")

                    with gr.TabItem("✅ Checklist"):
                        checklist_output = gr.Markdown(label="Checklist")

                    with gr.TabItem("👤 NER Entities"):
                        ner_output = gr.Markdown(label="NER")

                    with gr.TabItem("📅 Timeline"):
                        timeline_output = gr.Markdown(label="Timeline")

                # Download button
                with gr.Row():
                    download_btn = gr.Button(
                        "📥 Download Full Report",
                        variant="secondary",
                        size="lg",
                    )

                report_file = gr.File(
                    visible=False,
                    interactive=False,
                    label="Report Ready",
                )

        # Wire up events

        _PROCESSING_HTML = """
        <div id="processing-box" style="display:flex; align-items:center; gap:12px; padding:14px 20px; border-radius:10px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.22); margin-bottom:10px;">
            <div style="width:26px; height:26px; border:3px solid rgba(99,102,241,0.25); border-top:3px solid #6366f1; border-radius:50%; animation:cs-spin 0.8s linear infinite;"></div>
            <div>
                <div style="font-weight:600; color:#6366f1; font-size:0.95rem;">⏳ Processing your document...</div>
                <div id="cs-timer-text" style="font-size:0.85rem; color:#6b7280;">Elapsed: 0s</div>
            </div>
        </div>
        <style>@keyframes cs-spin { to { transform: rotate(360deg); } }</style>
        """

        _START_TIMER_JS = """
        () => {
            window._csStart = Date.now();
            if (window._csTimer) clearInterval(window._csTimer);
            window._csTimer = setInterval(() => {
                const el = document.getElementById('cs-timer-text');
                if (el) el.textContent = 'Elapsed: ' + Math.floor((Date.now() - window._csStart) / 1000) + 's';
            }, 500);
        }
        """

        _STOP_TIMER_JS = """
        () => {
            if (window._csTimer) { clearInterval(window._csTimer); window._csTimer = null; }
        }
        """

        def _show_processing():
            return gr.update(value=_PROCESSING_HTML, visible=True)

        def _hide_processing():
            return gr.update(value="", visible=False)

        analyse_btn.click(
            fn=_show_processing,
            inputs=[],
            outputs=[processing_status],
            js=_START_TIMER_JS,
        ).then(
            fn=run_analysis,
            inputs=[file_input, text_input, crime_type_dropdown, api_key_input],
            outputs=[summary_output, classification_output, checklist_output, ner_output, timeline_output, stats_output],
        ).then(
            fn=_hide_processing,
            inputs=[],
            outputs=[processing_status],
            js=_STOP_TIMER_JS,
        )

        regen_btn.click(
            fn=regenerate_checklist,
            inputs=[file_input, text_input, crime_type_dropdown, api_key_input],
            outputs=[checklist_output],
            show_progress="full",
        )

        download_btn.click(
            fn=generate_report,
            inputs=[file_input, text_input, api_key_input, summary_output, classification_output, checklist_output, ner_output, timeline_output],
            outputs=[report_file],
            show_progress="full",
        )

        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; padding: 15px 0;">
                <p style="font-size: 0.9rem; opacity: 0.7;"><strong>How to Use:</strong> Enter API Key → Upload/Paste Document → Click Analyse → View Results → Download Report</p>
            </div>
            """
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True,
    )
