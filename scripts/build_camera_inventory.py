"""Convert the police camera matrix sheet into config/camera_inventory.json (the grid's camera list).

Usage: python scripts/build_camera_inventory.py <sheet.xlsx> [--out config/camera_inventory.json]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.casework.cameras import normalise_row, read_xlsx_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=os.path.join(ROOT, "config", "camera_inventory.json"))
    a = ap.parse_args()
    rows, seen = [], set()
    for r in (normalise_row(x) for x in read_xlsx_rows(a.xlsx)):
        if "INVALID_IP" in r["issues"] or r["ip"] in seen or r["sheet_serial"] is None:
            print(f"skipped serial {r['sheet_serial']}: {r['issues'] or 'duplicate IP'}")
            continue
        seen.add(r["ip"])
        rows.append({"serial": r["sheet_serial"], "ps": r["ps"], "subdivision": r["sub"], "type": r["type"],
                     "location": r["location"], "lat": r["lat"], "lon": r["lon"], "ip": r["ip"],
                     "from": r["from"], "towards": r["towards"]})
    rows.sort(key=lambda r: r["serial"])
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(rows)} cameras to {a.out}")


if __name__ == "__main__":
    main()
