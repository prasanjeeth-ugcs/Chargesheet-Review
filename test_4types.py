"""Quick test: verify 4 crime types, checklists, rule detection, and all sample files."""
from processor import (
    list_crime_types, detect_crime_type_rules, load_checklists,
    extract_text, _rule_based_checklist, format_checklist_output,
    format_classification_output,
)

# 1. Check crime types
ct = list_crime_types()
print(f"Crime types: {len(ct)}")
for c in ct:
    print(f"  {c['key']}: {c['display_name']}")
print()

# 2. Check checklist sizes
cl = load_checklists()
for k, v in cl.items():
    print(f"{k}: {len(v['required_items'])} checklist items")
print()

# 3. Test rule detection on all 4 samples
samples = {
    "sample_theft_robbery.txt": "theft_robbery",
    "sample_assault_hurt.txt": "assault_hurt",
    "sample_cyber_fraud.txt": "cyber_fraud",
    "sample_ndps.txt": "ndps",
}

all_pass = True
for fname, expected in samples.items():
    text = open(fname, encoding="utf-8").read()
    rules = detect_crime_type_rules(text)
    if rules:
        detected = rules[0]["crime_key"]
        status = "PASS" if detected == expected else f"FAIL (got {detected})"
        if detected != expected:
            all_pass = False
        print(f"{fname}: {status} (score={rules[0]['score']})")
    else:
        print(f"{fname}: FAIL (no detection)")
        all_pass = False

print()

# 4. Test rule-based checklist on theft sample
text = open("sample_theft_robbery.txt", encoding="utf-8").read()
crime_info = cl["theft_robbery"]
result = _rule_based_checklist(text, crime_info["required_items"])
present = sum(1 for i in result["checklist"] if i["status"] == "present")
missing = sum(1 for i in result["checklist"] if i["status"] == "missing")
print(f"Theft checklist: {present} present, {missing} missing out of {len(result['checklist'])}")

# 5. Test formatting
output = format_checklist_output(result, "theft_robbery")
print(f"Formatted output length: {len(output)} chars")

print()
print("ALL TESTS PASSED!" if all_pass else "SOME TESTS FAILED!")
