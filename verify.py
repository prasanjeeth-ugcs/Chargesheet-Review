"""Quick verification script."""
import sys
print(f"Python: {sys.version}")

import gradio
import google.generativeai
import docx
import PyPDF2
import openai
print("All imports OK")

from processor import (
    load_checklists, list_crime_types, chunk_text,
    detect_crime_type_rules, extract_text,
)
print("Processor module OK")

ct = list_crime_types()
print(f"Crime types loaded: {len(ct)}")
for c in ct:
    print(f"  - {c['key']:25s} {c['display_name_en']}")

text = extract_text("sample_chargesheet_robbery.txt")
print(f"\nSample robbery doc: {len(text)} chars")

rules = detect_crime_type_rules(text)
print("\nRule-based detection:")
for r in rules:
    bar = "#" * r["score"]
    print(f"  {r['display_name']:50s} score={r['score']:3d}  {bar}")

text2 = extract_text("sample_chargesheet_cyber.txt")
print(f"\nSample cyber doc: {len(text2)} chars")

rules2 = detect_crime_type_rules(text2)
print("\nRule-based detection (cyber):")
for r in rules2:
    bar = "#" * r["score"]
    print(f"  {r['display_name']:50s} score={r['score']:3d}  {bar}")

print("\n✅ All checks passed!")
