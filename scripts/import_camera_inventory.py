"""Import the police camera matrix sheet into the casework camera inventory.

Usage: python scripts/import_camera_inventory.py <sheet.xlsx> [--db PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.casework.cameras import import_camera_sheet  # noqa: E402
from app.casework.schema import connect, init_casework_db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    path = init_casework_db(args.db)
    conn = connect(path)
    res = import_camera_sheet(conn, args.xlsx)
    print(f"{path}: {res['rows']} rows, {res['inserted']} inserted, {res['updated']} updated, "
          f"{res['skipped']} skipped")
    for f in res["flagged"]:
        print(f"  serial {f['serial']}: {', '.join(f['issues'])}")


if __name__ == "__main__":
    main()
