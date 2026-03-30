"""Quick test for new OCR corrections and NER filters."""
from processor import _apply_ocr_corrections, _repair_ocr_devanagari
import re

def test_ocr():
    tests = [
        ('बिामदगी', 'बरामदगी'),
        ('सम्पनि', 'सम्पत्ति'),
        ('सनगरेट', 'सिगरेट'),
        ('संभानवत', 'संभावित'),
        ('धिपुर', 'धनपुर'),
    ]
    print("=== Word-level OCR fixes ===")
    for inp, exp in tests:
        result = _apply_ocr_corrections(inp)
        status = "OK" if result == exp else "FAIL"
        print(f"  {inp} -> {result} (expect {exp}) [{status}]")

    # Phrase-level
    phrase_tests = [
        ('कु मार', 'कुमार'),
        ('कु न्द', 'कुन्द'),
        ('मृग कुमार', 'मनोज कुमार'),
        ('भवा नगर', 'भवानी नगर'),
        ('प्रकृ नत', 'प्रकृति'),
    ]
    print("\n=== Phrase-level OCR fixes ===")
    for inp, exp in phrase_tests:
        result = _apply_ocr_corrections(inp)
        status = "OK" if result == exp else "FAIL"
        print(f"  {inp} -> {result} (expect {exp}) [{status}]")

def test_filters():
    # Phone number filter
    _PHONE_RE = re.compile(
        r'^\s*(?:\+?91[\s-]*)?'
        r'(?:\d{2}xx\d{2}xx\d{2}|'
        r'\d{2}[xX*]+\d{2}[xX*]+\d{2,4}|'
        r'[6-9]\d{9})\s*$',
        re.IGNORECASE
    )
    phones = ['97xx43xx88', '98xx12xx34', '86xx78xx90', '95xx67xx21']
    money = ['₹45,000', '₹72,000', '₹12,500', '₹2,81,400']
    print("\n=== Phone number filter (MONETARY) ===")
    for p in phones:
        print(f"  {p}: filtered={bool(_PHONE_RE.match(p))} (should be True)")
    for m in money:
        print(f"  {m}: filtered={bool(_PHONE_RE.match(m))} (should be False)")

    # Ref number filter
    _REF_NUMBER_RE = re.compile(
        r'^\s*(?:'
        r'[A-Z]{2,}[/\-](?:[A-Z]{1,}[/\-])?\d{2,}[/\-]\d{2,}|'
        r'[A-Z]{2,}[/\-]\d{4}[/\-]\d+|'
        r'\d{3,4}[/\-]\d{4}'
        r')\s*$',
        re.IGNORECASE
    )
    refs = ['FSL/RC/2024/4478', 'MLG/2024/2187', 'CD/0214/2024', '0087/2022']
    orgs = ['CJM कोर्ट', 'BSNL', 'Airtel', 'स्टेट बैंक ऑफ इंडिया']
    print("\n=== Reference number filter (ORGANIZATION) ===")
    for r in refs:
        print(f"  {r}: filtered={bool(_REF_NUMBER_RE.match(r))} (should be True)")
    for o in orgs:
        print(f"  {o}: filtered={bool(_REF_NUMBER_RE.match(o))} (should be False)")

    # Plain number filter
    _PLAIN_NUMBER_RE = re.compile(r'^\s*\d{1,4}\s*$')
    nums = ['14', '22', '20', '18']
    locs = ['भवानी नगर', '17/बी', 'सुभाषनगर', 'नारायणपुर']
    print("\n=== Plain number filter (LOCATION) ===")
    for n in nums:
        print(f"  {n}: filtered={bool(_PLAIN_NUMBER_RE.match(n))} (should be True)")
    for l in locs:
        print(f"  {l}: filtered={bool(_PLAIN_NUMBER_RE.match(l))} (should be False)")

    # FIR number filter
    _FIR_NUMBER_RE = re.compile(r'^\s*0\d{2,4}\s*/\s*\d{4}\s*$')
    print("\n=== FIR number filter (DATE) ===")
    print(f"  0214/2024: filtered={bool(_FIR_NUMBER_RE.match('0214/2024'))} (should be True)")
    for d in ['12/09/2024', '11/09/2024', '14/03/1998']:
        print(f"  {d}: filtered={bool(_FIR_NUMBER_RE.match(d))} (should be False)")

if __name__ == '__main__':
    test_ocr()
    test_filters()
    print("\nAll tests completed!")
