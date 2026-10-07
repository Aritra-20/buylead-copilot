"""Supplier catalog loading + shared lookup tables (cities, synonyms)."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# City aliases buyers actually type -> canonical city. Includes cities that have no
# suppliers (e.g. Erode) so we can still capture the buyer's location correctly.
CITY_ALIASES = {
    "delhi": "Delhi", "new delhi": "Delhi", "dilli": "Delhi", "ncr": "Delhi",
    "noida": "Noida", "ghaziabad": "Ghaziabad", "gurgaon": "Gurugram", "gurugram": "Gurugram",
    "faridabad": "Faridabad", "ludhiana": "Ludhiana", "jaipur": "Jaipur", "ahmedabad": "Ahmedabad",
    "amdavad": "Ahmedabad", "rajkot": "Rajkot", "surat": "Surat", "mumbai": "Mumbai", "bombay": "Mumbai",
    "pune": "Pune", "chennai": "Chennai", "madras": "Chennai", "coimbatore": "Coimbatore",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "hyderabad": "Hyderabad", "kolkata": "Kolkata",
    "calcutta": "Kolkata", "indore": "Indore", "erode": "Erode", "lucknow": "Lucknow", "nagpur": "Nagpur",
}

CITY_STATE = {
    "Delhi": "Delhi", "Noida": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh", "Lucknow": "Uttar Pradesh",
    "Gurugram": "Haryana", "Faridabad": "Haryana", "Ludhiana": "Punjab", "Jaipur": "Rajasthan",
    "Ahmedabad": "Gujarat", "Rajkot": "Gujarat", "Surat": "Gujarat", "Mumbai": "Maharashtra",
    "Pune": "Maharashtra", "Nagpur": "Maharashtra", "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Erode": "Tamil Nadu", "Bengaluru": "Karnataka", "Hyderabad": "Telangana", "Kolkata": "West Bengal",
    "Indore": "Madhya Pradesh",
}
# Delhi NCR buyers are happy with NCR suppliers
NCR = {"Delhi", "Noida", "Ghaziabad", "Gurugram", "Faridabad"}


@lru_cache(maxsize=1)
def load_suppliers() -> list[dict]:
    rows = []
    with (DATA_DIR / "suppliers.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["price_min_inr"] = float(r["price_min_inr"])
            r["price_max_inr"] = float(r["price_max_inr"])
            r["moq"] = int(r["moq"])
            r["rating"] = float(r["rating"])
            r["avg_response_hrs"] = int(r["avg_response_hrs"])
            r["gst_verified"] = r["gst_verified"] == "True"
            rows.append(r)
    return rows


@lru_cache(maxsize=1)
def products() -> dict[str, dict]:
    """Catalog product name -> {category, unit, keywords}."""
    out: dict[str, dict] = {}
    for r in load_suppliers():
        out.setdefault(r["product"], {"category": r["category"], "unit": r["unit"], "keywords": r["keywords"]})
    return out
