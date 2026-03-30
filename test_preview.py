"""
Preview test: Run _canonicalize_entities on the actual NER output from the user's
document to verify all filters work correctly.
"""
from processor import _canonicalize_entities, _apply_ocr_corrections, _repair_ocr_devanagari
import json

# Simulated NER entities matching the user's actual output
raw_entities = [
    # ACCUSED
    {"text": "बिट्टू", "type": "ACCUSED"},
    {"text": "विकास कुमार", "type": "ACCUSED"},
    {"text": "संजय", "type": "ACCUSED"},
    {"text": "सज्जू", "type": "ACCUSED"},
    {"text": "मुन्ना", "type": "ACCUSED"},
    {"text": "रामदेव प्रसाद", "type": "ACCUSED"},
    {"text": "काला", "type": "ACCUSED"},
    
    # WITNESS
    {"text": "सुरेश यादव", "type": "WITNESS"},
    {"text": "महेश सिंह", "type": "WITNESS"},
    {"text": "रामनिकशोर पटेल", "type": "WITNESS"},
    {"text": "अरविन्द कुमार मिश्रा", "type": "WITNESS"},
    {"text": "राजेंद्र प्रसाद", "type": "WITNESS"},
    {"text": "दीनदयाल शर्मा", "type": "WITNESS"},
    
    # OFFICER
    {"text": "राम चंद्र तिवारी", "type": "OFFICER"},
    {"text": "रवि शंकर यादव", "type": "OFFICER"},
    {"text": "धर्मेंद्र सिंह", "type": "OFFICER"},
    {"text": "श्यामलाल वर्मा", "type": "OFFICER"},
    {"text": "पुलिस अवर निरीक्षक", "type": "OFFICER"},
    {"text": "थानाध्यक्ष, भवानी नगर", "type": "OFFICER"},
    
    # DOCTOR
    {"text": "डॉ. राजेश कुमार वर्मा", "type": "DOCTOR"},
    {"text": "डॉ. अनुराग सिंह", "type": "DOCTOR"},
    
    # PERSON
    {"text": "मनोज कुमार गुप्ता", "type": "PERSON"},
    {"text": "रामसेवक गुप्ता", "type": "PERSON"},
    {"text": "सुमन गुप्ता", "type": "PERSON"},
    {"text": "हरिशंकर यादव", "type": "PERSON"},
    {"text": "विजय बहादुर सिंह", "type": "PERSON"},
    {"text": "लालता प्रसाद पटेल", "type": "PERSON"},
    {"text": "रामप्रसाद मिश्रा", "type": "PERSON"},
    {"text": "जगदीश प्रसाद", "type": "PERSON"},
    {"text": "शिवशंकर प्रसाद", "type": "PERSON"},
    {"text": "हरि प्रसाद", "type": "PERSON"},
    {"text": "ओमप्रकाश केसरवानी", "type": "PERSON"},
    {"text": "गीता देवी", "type": "PERSON"},
    {"text": "संतोष कुमार", "type": "PERSON"},
    {"text": "रमेश कुमार", "type": "PERSON"},
    
    # DATE — includes FIR number that should be filtered
    {"text": "0214/2024", "type": "DATE"},      # <-- FIR number, NOT a date
    {"text": "12/09/2024", "type": "DATE"},
    {"text": "11/09/2024", "type": "DATE"},
    {"text": "13/09/2024", "type": "DATE"},
    {"text": "20-25 अगस्त 2024", "type": "DATE"},
    {"text": "14/09/2024", "type": "DATE"},
    {"text": "18/09/2024", "type": "DATE"},
    {"text": "14/03/1998", "type": "DATE"},
    {"text": "19/09/2024", "type": "DATE"},
    {"text": "20/09/2024", "type": "DATE"},
    {"text": "22/09/2024", "type": "DATE"},
    {"text": "05/10/2024", "type": "DATE"},
    {"text": "25/09/2024", "type": "DATE"},
    {"text": "27/09/2024", "type": "DATE"},
    {"text": "10/10/2024", "type": "DATE"},
    {"text": "15/10/2024", "type": "DATE"},
    {"text": "28/09/2024", "type": "DATE"},
    {"text": "09/10/2024", "type": "DATE"},
    {"text": "10/09/2024", "type": "DATE"},
    {"text": "15/09/2024", "type": "DATE"},
    {"text": "17/09/2024", "type": "DATE"},
    {"text": "21/09/2024", "type": "DATE"},
    {"text": "25/10/2024", "type": "DATE"},
    {"text": "22/10/2024", "type": "DATE"},
    {"text": "23/09/2024", "type": "DATE"},
    {"text": "26/09/2024", "type": "DATE"},
    
    # LOCATION — includes plain numbers that should be filtered
    {"text": "भवानी नगर", "type": "LOCATION"},
    {"text": "धनपुर-मझगाँव", "type": "LOCATION"},
    {"text": "17/बी", "type": "LOCATION"},
    {"text": "सुभाषनगर", "type": "LOCATION"},
    {"text": "14", "type": "LOCATION"},          # <-- plain number
    {"text": "22", "type": "LOCATION"},          # <-- plain number
    {"text": "रामपुर", "type": "LOCATION"},
    {"text": "नारायणपुर", "type": "LOCATION"},
    {"text": "सिकंदरपुर", "type": "LOCATION"},
    {"text": "धनपुर बाजार", "type": "LOCATION"},
    {"text": "20", "type": "LOCATION"},          # <-- plain number
    {"text": "18", "type": "LOCATION"},          # <-- plain number
    
    # LEGAL_SECTION
    {"text": "BNSS 180", "type": "LEGAL_SECTION"},
    
    # ORGANIZATION — includes ref numbers that should be filtered
    {"text": "FSL/RC/2024/4478", "type": "ORGANIZATION"},   # <-- ref number
    {"text": "जिला चिकित्सालय", "type": "ORGANIZATION"},
    {"text": "MLG/2024/2187", "type": "ORGANIZATION"},       # <-- ref number
    {"text": "नगर पंचायत, भवानी नगर", "type": "ORGANIZATION"},
    {"text": "Hikvision", "type": "ORGANIZATION"},
    {"text": "रीजनल कंप्यूटर फॉरेंसिक लेबोरेटरी", "type": "ORGANIZATION"},
    {"text": "कोतवाली मझगाँव", "type": "ORGANIZATION"},
    {"text": "CJM कोर्ट", "type": "ORGANIZATION"},
    {"text": "CD/0214/2024", "type": "ORGANIZATION"},        # <-- ref number
    {"text": "Airtel", "type": "ORGANIZATION"},
    {"text": "BSNL", "type": "ORGANIZATION"},
    {"text": "0087/2022", "type": "ORGANIZATION"},           # <-- ref number
    {"text": "जिला कारागार", "type": "ORGANIZATION"},
    {"text": "ACJM-II", "type": "ORGANIZATION"},
    {"text": "स्टेट बैंक ऑफ इंडिया", "type": "ORGANIZATION"},
    
    # LANDMARK
    {"text": "सरकारी प्राथमिक विद्यालय", "type": "LANDMARK"},
    {"text": "स्टेट हाईवे", "type": "LANDMARK"},
    {"text": "रेलवे स्टेशन, भवानी नगर", "type": "LANDMARK"},
    {"text": "धनपुर बस अड्डा", "type": "LANDMARK"},
    {"text": "चाय की दुकान, स्टेशन रोड", "type": "LANDMARK"},
    
    # EVIDENCE — includes OCR error "सनगरेट"
    {"text": "Samsung Galaxy S21", "type": "EVIDENCE"},
    {"text": "महिला की सोने की नथ", "type": "EVIDENCE"},
    {"text": "HP brand Laptop", "type": "EVIDENCE"},
    {"text": "टूटी हुई कुंडी का टुकड़ा", "type": "EVIDENCE"},
    {"text": "काले रंग का कपड़ा", "type": "EVIDENCE"},
    {"text": "गोल्ड फ्लेक सनगरेट की खाली डिब्बी", "type": "EVIDENCE"},  # <-- OCR error
    {"text": "स्क्रूड्राइवर", "type": "EVIDENCE"},
    {"text": "फिंगर प्रिंट लिफ्ट", "type": "EVIDENCE"},
    {"text": "जूते के निशान की कास्ट", "type": "EVIDENCE"},
    {"text": "Pulsar", "type": "EVIDENCE"},
    {"text": "Hero Splendor", "type": "EVIDENCE"},
    {"text": "DVR", "type": "EVIDENCE"},
    {"text": "32GB Pen Drive", "type": "EVIDENCE"},
    {"text": "एयरटेल सिम", "type": "EVIDENCE"},
    {"text": "सोने की चेन", "type": "EVIDENCE"},
    {"text": "सोने की अंगूठी", "type": "EVIDENCE"},
    {"text": "चाँदी के पायल", "type": "EVIDENCE"},
    {"text": "HP Laptop", "type": "EVIDENCE"},
    {"text": "5CG1xxxxxx", "type": "EVIDENCE"},
    {"text": "चाकू", "type": "EVIDENCE"},
    {"text": "Redmi मोबाइल", "type": "EVIDENCE"},
    
    # MONETARY — includes phone numbers that should be filtered
    {"text": "97xx43xx88", "type": "MONETARY"},     # <-- phone number
    {"text": "₹45,000", "type": "MONETARY"},
    {"text": "₹72,000", "type": "MONETARY"},
    {"text": "₹36,000", "type": "MONETARY"},
    {"text": "₹8,400", "type": "MONETARY"},
    {"text": "₹18,000", "type": "MONETARY"},
    {"text": "₹2,81,400", "type": "MONETARY"},
    {"text": "98xx12xx34", "type": "MONETARY"},     # <-- phone number
    {"text": "86xx78xx90", "type": "MONETARY"},     # <-- phone number
    {"text": "95xx67xx21", "type": "MONETARY"},     # <-- phone number
    {"text": "91xx34xx56", "type": "MONETARY"},     # <-- phone number
    {"text": "88xx91xx07", "type": "MONETARY"},     # <-- phone number
    {"text": "79xx23xx11", "type": "MONETARY"},     # <-- phone number
    {"text": "94xx55xx82", "type": "MONETARY"},     # <-- phone number
    {"text": "₹12,500", "type": "MONETARY"},
    {"text": "76xx88xx33", "type": "MONETARY"},     # <-- phone number
    {"text": "84xx12xx77", "type": "MONETARY"},     # <-- phone number
    {"text": "87xx44xx09", "type": "MONETARY"},     # <-- phone number
    {"text": "₹4,500", "type": "MONETARY"},
    {"text": "₹500", "type": "MONETARY"},
]

print(f"Input: {len(raw_entities)} entities")
print("=" * 80)

# Run canonicalization (which includes all filtering)
result = _canonicalize_entities(raw_entities)

print(f"\nOutput: {len(result)} entities (filtered {len(raw_entities) - len(result)})")
print("=" * 80)

# Group by type and display
by_type = {}
for e in result:
    by_type.setdefault(e["type"], []).append(e["text"])

for etype in ["ACCUSED", "WITNESS", "OFFICER", "DOCTOR", "PERSON",
              "DATE", "LOCATION", "LEGAL_SECTION", "ORGANIZATION",
              "LANDMARK", "EVIDENCE", "MONETARY"]:
    items = by_type.get(etype, [])
    if items:
        print(f"\n{etype} ({len(items)}):")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")

# Check specific filters worked
print("\n" + "=" * 80)
print("FILTER VERIFICATION:")
print("=" * 80)

# Check phone numbers removed from MONETARY
monetary = by_type.get("MONETARY", [])
phone_in_monetary = [m for m in monetary if 'xx' in m]
if phone_in_monetary:
    print(f"❌ FAIL: Phone numbers still in MONETARY: {phone_in_monetary}")
else:
    print("✅ Phone numbers filtered from MONETARY")

# Check ref numbers removed from ORGANIZATION  
orgs = by_type.get("ORGANIZATION", [])
ref_in_org = [o for o in orgs if '/' in o and any(c.isdigit() for c in o) and not any('\u0900' <= c <= '\u097F' for c in o)]
if ref_in_org:
    print(f"❌ FAIL: Reference numbers still in ORGANIZATION: {ref_in_org}")
else:
    print("✅ Reference numbers filtered from ORGANIZATION")

# Check plain numbers removed from LOCATION
locs = by_type.get("LOCATION", [])
num_in_loc = [l for l in locs if l.strip().isdigit()]
if num_in_loc:
    print(f"❌ FAIL: Plain numbers still in LOCATION: {num_in_loc}")
else:
    print("✅ Plain numbers filtered from LOCATION")

# Check FIR number removed from DATE
dates = by_type.get("DATE", [])
if "0214/2024" in dates:
    print("❌ FAIL: FIR number '0214/2024' still in DATE")
else:
    print("✅ FIR number '0214/2024' filtered from DATE")

# Check OCR fix in EVIDENCE
evidence = by_type.get("EVIDENCE", [])
ocr_fixed = any("सिगरेट" in e for e in evidence)
ocr_bad = any("सनगरेट" in e for e in evidence)
if ocr_fixed and not ocr_bad:
    print("✅ OCR fix: सनगरेट → सिगरेट in EVIDENCE")
elif ocr_bad:
    print("❌ FAIL: सनगरेट still not fixed in EVIDENCE")
else:
    print("⚠️ सिगरेट not found in EVIDENCE (may have been deduped)")

# Check HP Laptop dedup
hp_laptops = [e for e in evidence if 'Laptop' in e or 'laptop' in e or 'लैपटॉप' in e]
if len(hp_laptops) <= 1:
    print(f"✅ HP Laptop deduped: {hp_laptops}")
else:
    print(f"⚠️ Multiple laptop entries: {hp_laptops}")

print(f"\nTotal entity count: {len(result)} (was {len(raw_entities)})")
