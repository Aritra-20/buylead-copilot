"""Generate a SYNTHETIC B2B supplier catalog for BuyLead Copilot.

All supplier names, prices and ratings are fictional and generated with a fixed
random seed so results are reproducible. Product families mirror common Indian
B2B marketplace categories (fasteners, packaging, PPE, electricals, etc.).

Usage:  python scripts/generate_catalog.py
"""
import csv
import random
from pathlib import Path

random.seed(42)

# (category, product, [spec variants], unit, (price_lo, price_hi), (moq_lo, moq_hi), search keywords)
PRODUCTS = [
    ("Fasteners", "Stainless Steel Hex Bolt", ["M6 SS304", "M8 SS304", "M10 SS304", "M8 SS316", "M12 SS316"], "piece", (3, 28), (100, 2000), "ss bolt hex bolt nut bolt fastener stainless"),
    ("Fasteners", "MS Hex Nut", ["M8 zinc plated", "M10 zinc plated", "M12 black"], "piece", (1, 6), (500, 5000), "nut ms nut hex nut fastener"),
    ("Fasteners", "Self Drilling Screw", ["10x25mm", "12x50mm"], "piece", (1, 4), (1000, 10000), "screw roofing screw self drilling"),
    ("Packaging", "Corrugated Box", ["3 ply 12x10x8 inch", "5 ply 18x12x12 inch", "7 ply 24x18x18 inch"], "piece", (8, 65), (200, 2000), "carton box corrugated box packaging gatta peti"),
    ("Packaging", "BOPP Packing Tape", ["48mm x 65m brown", "72mm x 100m transparent"], "roll", (25, 90), (100, 1000), "tape bopp tape packing tape cello tape"),
    ("Packaging", "Stretch Wrap Film", ["18 inch 23 micron", "20 inch 25 micron"], "kg", (140, 190), (50, 500), "stretch film wrap film pallet wrap"),
    ("Safety & PPE", "Industrial Safety Helmet", ["ISI marked ratchet", "ISI marked pin lock"], "piece", (90, 320), (50, 500), "helmet safety helmet hard hat ppe"),
    ("Safety & PPE", "Nitrile Gloves", ["powder free medium", "powder free large"], "box", (280, 520), (20, 200), "gloves nitrile gloves hand gloves ppe"),
    ("Safety & PPE", "Safety Shoes", ["steel toe size 6-11", "PU sole steel toe"], "pair", (450, 1400), (20, 200), "safety shoes boots ppe steel toe"),
    ("Pipes & Fittings", "PVC Pipe", ["1 inch 6kg", "2 inch 6kg", "4 inch 4kg"], "metre", (45, 380), (100, 1000), "pvc pipe plastic pipe water pipe"),
    ("Pipes & Fittings", "GI Pipe", ["1 inch medium class", "2 inch heavy class"], "metre", (180, 620), (50, 500), "gi pipe galvanised pipe iron pipe"),
    ("Electrical", "Copper House Wire", ["1.5 sq mm FR 90m", "2.5 sq mm FRLS 90m", "4 sq mm FRLS 90m"], "coil", (1100, 4200), (10, 100), "wire copper wire house wire cable"),
    ("Electrical", "MCB", ["16A single pole", "32A double pole", "63A four pole"], "piece", (120, 1800), (20, 200), "mcb circuit breaker switchgear"),
    ("Electrical", "LED Flood Light", ["50W IP65", "100W IP66", "200W IP66"], "piece", (450, 3200), (10, 100), "led flood light outdoor light"),
    ("Solar", "Mono PERC Solar Panel", ["540W", "445W", "335W"], "piece", (9500, 16500), (5, 50), "solar panel solar plate pv module"),
    ("Textiles", "Cotton Fabric", ["60s poplin white", "40s cambric dyed", "denim 12oz"], "metre", (55, 220), (500, 5000), "cotton fabric kapda cloth textile"),
    ("Chemicals", "Caustic Soda Flakes", ["98% purity 50kg bag"], "kg", (38, 62), (500, 10000), "caustic soda sodium hydroxide flakes"),
    ("Agriculture", "Basmati Rice", ["1121 steam sella", "1509 golden sella", "pusa raw"], "kg", (62, 115), (1000, 25000), "basmati rice chawal rice"),
    ("Office Furniture", "Ergonomic Office Chair", ["mesh back with headrest", "high back leatherette"], "piece", (3200, 11500), (5, 50), "office chair chair revolving chair furniture"),
    ("Machinery", "Mini Dal Mill Machine", ["1 HP 100 kg/hr", "3 HP 300 kg/hr"], "unit", (45000, 185000), (1, 5), "dal mill machine pulse machine food processing"),
]

CITIES = [
    ("Delhi", "Delhi"), ("Noida", "Uttar Pradesh"), ("Ghaziabad", "Uttar Pradesh"), ("Gurugram", "Haryana"),
    ("Faridabad", "Haryana"), ("Ludhiana", "Punjab"), ("Jaipur", "Rajasthan"), ("Ahmedabad", "Gujarat"),
    ("Rajkot", "Gujarat"), ("Surat", "Gujarat"), ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"),
    ("Chennai", "Tamil Nadu"), ("Coimbatore", "Tamil Nadu"), ("Bengaluru", "Karnataka"), ("Hyderabad", "Telangana"),
    ("Kolkata", "West Bengal"), ("Indore", "Madhya Pradesh"),
]

NAME_A = ["Shree", "Om", "Sai", "Balaji", "Ganesh", "Krishna", "Laxmi", "Jai", "Maa", "Royal", "Prime", "Apex", "Vardhman", "Unique", "National"]
NAME_B = ["Industries", "Traders", "Enterprises", "Corporation", "Impex", "Udyog", "Sales", "Exports", "Agencies", "Manufacturing Co."]


def main() -> None:
    rows = []
    sid = 1000
    used = set()
    for cat, prod, specs, unit, (plo, phi), (mlo, mhi), kw in PRODUCTS:
        n_suppliers = random.randint(9, 14)
        for _ in range(n_suppliers):
            sid += 1
            while True:
                name = f"{random.choice(NAME_A)} {random.choice(NAME_B)}"
                city, state = random.choice(CITIES)
                if (name, city) not in used:
                    used.add((name, city))
                    break
            spec = random.choice(specs)
            lo = round(random.uniform(plo, phi * 0.85), 2)
            hi = round(lo * random.uniform(1.08, 1.35), 2)
            moq = int(round(random.uniform(mlo, mhi), -1)) or mlo
            rows.append({
                "supplier_id": f"S{sid}",
                "supplier_name": f"{name} ({city})",
                "city": city,
                "state": state,
                "category": cat,
                "product": prod,
                "spec": spec,
                "unit": unit,
                "price_min_inr": lo,
                "price_max_inr": hi,
                "moq": moq,
                "gst_verified": random.random() < 0.78,
                "rating": round(random.uniform(3.4, 4.9), 1),
                "avg_response_hrs": random.choice([1, 2, 4, 6, 12, 24]),
                "keywords": kw,
            })
    out = Path(__file__).resolve().parents[1] / "data" / "suppliers.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic suppliers -> {out}")


if __name__ == "__main__":
    main()
