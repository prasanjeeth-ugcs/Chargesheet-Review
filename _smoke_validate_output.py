from processor import process_chargesheet, format_timeline_output

with open("sample_theft_robbery.txt", "r", encoding="utf-8") as file_handle:
    text = file_handle.read()

out = process_chargesheet(text)
summary = out.get("summary", "")

print("SUMMARY_VALIDATED_COUNT=", summary.count("**Validated Legal Sections:**"))
print("HAS_VALIDATED_LINE=", "**Validated Legal Sections:**" in summary)

html = format_timeline_output(
    out.get("ner_entities", []),
    summary,
    out.get("preprocessed_text", text),
)
print("TIMELINE_HAS_FROM_TO=", ("From <strong" in html and "To <strong" in html))
print("TIMELINE_HAS_PRIOR_LABEL=", ("Prior Criminal Record" in html or "पूर्व आपराधिक" in html))
