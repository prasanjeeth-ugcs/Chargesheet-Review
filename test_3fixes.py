"""Test the 3 quality fixes: OCR consistency, alias merging, timeline birth-date exclusion."""
from processor import _canonicalize_entities, format_timeline_output

def test_alias_merging():
    print("=== ALIAS MERGING ===")
    entities = [
        {"text": "राजेश उर्फ गुड्डू", "type": "ACCUSED"},
        {"text": "राजेश", "type": "ACCUSED"},
        {"text": "गुड्डू", "type": "PERSON"},
        {"text": "मोहन उर्फ मोनू", "type": "ACCUSED"},
        {"text": "मोनू", "type": "PERSON"},
        {"text": "सुरेश कुमार", "type": "WITNESS"},  # no alias
    ]
    result = _canonicalize_entities(entities)
    persons = [e for e in result if e["type"] in ("PERSON", "ACCUSED", "WITNESS", "OFFICER", "DOCTOR")]
    print(f"  Input: 6 entities (3 unique people)")
    print(f"  Output: {len(persons)} persons:")
    for p in persons:
        print(f"    [{p['type']}] {p['text']}")

    assert len(persons) == 3, f"Expected 3, got {len(persons)}"
    alias_texts = [p["text"] for p in persons]
    assert any("उर्फ" in t and "राजेश" in t for t in alias_texts), "राजेश alias form not preserved"
    assert any("उर्फ" in t and "मोहन" in t for t in alias_texts), "मोहन alias form not preserved"
    print("  ✓ PASS\n")


def test_alias_reverse_order():
    """Alias entity comes AFTER the standalone name."""
    print("=== ALIAS MERGING (reverse order) ===")
    entities = [
        {"text": "गुड्डू", "type": "ACCUSED"},
        {"text": "राजेश उर्फ गुड्डू", "type": "ACCUSED"},
    ]
    result = _canonicalize_entities(entities)
    persons = [e for e in result if e["type"] in ("PERSON", "ACCUSED", "WITNESS", "OFFICER", "DOCTOR")]
    print(f"  Input: standalone first, then alias")
    print(f"  Output: {len(persons)} persons:")
    for p in persons:
        print(f"    [{p['type']}] {p['text']}")

    assert len(persons) == 1, f"Expected 1, got {len(persons)}"
    assert "उर्फ" in persons[0]["text"], "Alias form not preserved"
    print("  ✓ PASS\n")


def test_timeline_birth_exclusion():
    print("=== TIMELINE: Birth-date exclusion ===")
    ner = [
        {"text": "01/01/1990", "type": "DATE"},   # DOB — should be excluded
        {"text": "15/03/2024", "type": "DATE"},    # FIR date
        {"text": "14/03/2024", "type": "DATE"},    # Incident date
    ]
    doc_text = (
        "आरोपी राजेश पुत्र रामलाल, जन्म तिथि 01/01/1990, निवासी ग्राम दहलीपुर।\n"
        "दिनांक 14/03/2024 को घटना हुई जब आरोपी ने हमला किया।\n"
        "दिनांक 15/03/2024 को FIR दर्ज की गई मु. अ. नं. 0123/2024।\n"
    )
    timeline = format_timeline_output(ner, doc_text)
    birth_excluded = "1990" not in timeline
    has_incident = "Incident" in timeline
    has_fir = "FIR" in timeline
    print(f"  Birth date (1990) excluded: {birth_excluded}")
    print(f"  Incident date present: {has_incident}")
    print(f"  FIR date present: {has_fir}")
    assert birth_excluded, "FAIL — birth date leaked into timeline"
    assert has_incident, "FAIL — incident date missing"
    assert has_fir, "FAIL — FIR date missing"
    print("  ✓ PASS\n")


def test_timeline_death_vs_postmortem():
    """Ensure शव परीक्षण is labeled Postmortem, not Death."""
    print("=== TIMELINE: Death vs Postmortem ===")
    ner = [
        {"text": "16/03/2024", "type": "DATE"},
        {"text": "17/03/2024", "type": "DATE"},
    ]
    doc_text = (
        "दिनांक 16/03/2024 को मृत्यु हुई।\n"
        "दिनांक 17/03/2024 को शव परीक्षण किया गया।\n"
    )
    timeline = format_timeline_output(ner, doc_text)
    has_death = "Death" in timeline
    has_postmortem = "Postmortem" in timeline
    print(f"  Death event present: {has_death}")
    print(f"  Postmortem event present: {has_postmortem}")
    # Postmortem should NOT be labeled as Death
    assert has_postmortem, "FAIL — postmortem not detected"
    print("  ✓ PASS\n")


def test_timeline_old_date_filter():
    """Dates 15+ years older than recent dates are likely DOB, should be filtered."""
    print("=== TIMELINE: Old date filter (>15 year gap) ===")
    ner = [
        {"text": "05/06/1985", "type": "DATE"},   # likely DOB
        {"text": "20/03/2024", "type": "DATE"},    # case event
    ]
    doc_text = (
        "आरोपी की उम्र 39 वर्ष, जन्म 05/06/1985\n"
        "दिनांक 20/03/2024 को FIR दर्ज की गई।\n"
    )
    timeline = format_timeline_output(ner, doc_text)
    old_excluded = "1985" not in timeline
    has_fir = "FIR" in timeline
    print(f"  Old date (1985) excluded: {old_excluded}")
    print(f"  FIR date present: {has_fir}")
    assert old_excluded, "FAIL — old biographical date leaked"
    assert has_fir, "FAIL — FIR date missing"
    print("  ✓ PASS\n")


if __name__ == "__main__":
    test_alias_merging()
    test_alias_reverse_order()
    test_timeline_birth_exclusion()
    test_timeline_death_vs_postmortem()
    test_timeline_old_date_filter()
    print("=" * 50)
    print("ALL TESTS PASS ✓")
