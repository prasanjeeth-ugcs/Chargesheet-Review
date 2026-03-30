from processor import (
    _is_relationship_phrase,
    _dedup_legal_sections,
    _filter_procedural_sections,
    _is_title_only_entity,
    _canonicalize_entities,
    _upsert_validated_sections_line,
    format_timeline_output,
)


def test_relationship_phrase_detection():
    assert _is_relationship_phrase("मो0 इरफान की मौसी") is True
    assert _is_relationship_phrase("राजेश कुमार") is False


def test_legal_dedup_specificity():
    inp = ["BNS 103", "BNS 103(1)", "BNSS 180", "BNSS 180"]
    out = _dedup_legal_sections(inp)
    assert "BNS 103(1)" in out
    assert "BNS 103" not in out
    assert out.count("BNSS 180") == 1


def test_filter_procedural_sections():
    sections = ["BNS 103", "380/457 भा0 द0 वि0", "धारा 63(4)"]
    validated = ["BNS 103(1)"]
    out = _filter_procedural_sections(sections, validated, "")
    assert any("BNS 103" in s for s in out)
    assert any("380/457" in s for s in out)


def test_title_only_officer_detection():
    assert _is_title_only_entity("पु0 अ0नि0 सह थाना प्रभारी सरायकेला") is True
    assert _is_title_only_entity("सतीश वर्णवाल") is False


def test_timeline_header_excludes_prior_anchor():
    ner = [
        {"text": "01/01/2022", "type": "DATE"},
        {"text": "10/09/2024", "type": "DATE"},
        {"text": "20/10/2024", "type": "DATE"},
    ]
    doc = (
        "आपराधिक इतिहास: दिनांक 01/01/2022 काण्ड सं0 पुराना मामला।\n"
        "दिनांक 10/09/2024 को घटना हुई।\n"
        "दिनांक 20/10/2024 को चार्जशीट दाखिल हुई।\n"
    )
    html = format_timeline_output(ner, doc, doc)
    assert "10 Sep 2024" in html and "20 Oct 2024" in html


def test_possessive_token_false_positive_guard():
    ents = [{"text": "विकास कुमार", "type": "ACCUSED"}]
    out = _canonicalize_entities(ents)
    assert len(out) == 1
    assert out[0]["type"] == "ACCUSED"


def test_single_validated_sections_line():
    summary = "Header\n\n**Validated Legal Sections:** BNS 103"
    out = _upsert_validated_sections_line(summary, ["BNS 103", "BNS 103(1)"])
    assert out.count("**Validated Legal Sections:**") == 1
    assert "BNS 103(1)" in out


if __name__ == "__main__":
    test_relationship_phrase_detection()
    test_legal_dedup_specificity()
    test_filter_procedural_sections()
    test_title_only_officer_detection()
    test_timeline_header_excludes_prior_anchor()
    test_possessive_token_false_positive_guard()
    test_single_validated_sections_line()
    print("ALL ISSUE FIX TESTS PASS")
