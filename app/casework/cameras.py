"""Camera inventory: import from the police camera matrix sheet and find cameras near a point."""

import json
import math
import os
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_IP = re.compile(r"^(\d{1,3})(\.\d{1,3}){3}$")
HEADERS = {"S.No": "serial", "PS Name": "ps", "Sub Division Name": "sub", "Type of Cameras": "type",
           "Camera location": "location", "Latitude": "lat", "Longitude": "lon", "IP address": "ip",
           "Fromwords": "from", "Towards": "towards"}


def read_xlsx_rows(path: str) -> List[Dict[str, str]]:
    """Minimal first-sheet reader (stdlib only) returning rows keyed by header text."""
    with zipfile.ZipFile(path) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(_NS + "t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(_NS + "si")
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    grid: List[Dict[str, str]] = []
    for row in sheet.iter(_NS + "row"):
        cells: Dict[str, str] = {}
        for c in row.findall(_NS + "c"):
            v = c.find(_NS + "v")
            if v is None:
                continue
            col = re.match(r"[A-Z]+", c.get("r")).group()
            cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        grid.append(cells)
    head = {col: HEADERS[name.strip()] for col, name in grid[0].items() if name.strip() in HEADERS}
    missing = set(HEADERS.values()) - set(head.values())
    if missing:
        raise ValueError(f"Camera sheet is missing columns: {sorted(missing)}")
    return [{head[c]: (v or "").strip() for c, v in r.items() if c in head} for r in grid[1:] if r]


def _num(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def normalise_row(row: Dict[str, str]) -> Dict[str, Any]:
    issues: List[str] = []
    lat, lon = _num(row.get("lat", "")), _num(row.get("lon", ""))
    if lat is None or lon is None:
        issues.append("MISSING_COORDINATES")
        lat = lon = None
    elif not (-90 <= lat <= 90 and -180 <= lon <= 180):
        issues.append("COORDINATES_OUT_OF_RANGE")
        lat = lon = None
    ip = row.get("ip", "")
    if not _IP.match(ip) or any(int(p) > 255 for p in ip.split(".")):
        issues.append("INVALID_IP")
    towards = row.get("towards", "")
    if re.fullmatch(r"(?i)towards", towards):
        towards = ""
    serial = row.get("serial", "")
    return {
        "sheet_serial": int(float(serial)) if _num(serial) is not None else None,
        "ip": ip, "type": row.get("type", "").upper() or "UNKNOWN",
        "location": re.sub(r"\s+", " ", row.get("location", "")),
        "ps": row.get("ps", ""), "sub": row.get("sub", ""), "lat": lat, "lon": lon,
        "from": row.get("from", ""), "towards": towards, "issues": issues,
    }


def import_camera_sheet(conn: sqlite3.Connection, xlsx_path: str) -> Dict[str, Any]:
    """Upsert every sheet row by IP address. Safe to re-run; reports rows it could not trust."""
    rows = [normalise_row(r) for r in read_xlsx_rows(xlsx_path)]
    now, src = time.time(), os.path.basename(xlsx_path)
    inserted = updated = skipped = 0
    seen: Dict[str, int] = {}
    flagged: List[Dict[str, Any]] = []
    for r in rows:
        if "INVALID_IP" in r["issues"]:
            skipped += 1
            flagged.append({"serial": r["sheet_serial"], "issues": r["issues"]})
            continue
        if r["ip"] in seen:
            r["issues"].append(f"DUPLICATE_IP_OF_SERIAL_{seen[r['ip']]}")
            skipped += 1
            flagged.append({"serial": r["sheet_serial"], "issues": r["issues"]})
            continue
        seen[r["ip"]] = r["sheet_serial"]
        cid = f"MTX-{r['sheet_serial']:04d}" if r["sheet_serial"] is not None else "MTX-IP-" + r["ip"].replace(".", "-")
        exists = conn.execute("SELECT 1 FROM cw_cameras WHERE ip_address = ?", (r["ip"],)).fetchone()
        vals = (r["sheet_serial"], r["type"], r["location"], r["ps"], r["sub"], r["lat"], r["lon"],
                r["from"] or None, r["towards"] or None, json.dumps(r["issues"]), src, now)
        if exists:
            conn.execute(
                "UPDATE cw_cameras SET sheet_serial=?, camera_type=?, location=?, police_station=?, "
                "sub_division=?, latitude=?, longitude=?, from_side=?, towards=?, data_issues=?, "
                "source_file=?, imported_at=? WHERE ip_address=?", vals + (r["ip"],))
            updated += 1
        else:
            conn.execute(
                "INSERT INTO cw_cameras (camera_id, ip_address, sheet_serial, camera_type, location, "
                "police_station, sub_division, latitude, longitude, from_side, towards, data_issues, "
                "source_file, imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid, r["ip"]) + vals)
            inserted += 1
        if r["issues"]:
            flagged.append({"serial": r["sheet_serial"], "issues": r["issues"]})
    conn.commit()
    return {"rows": len(rows), "inserted": inserted, "updated": updated,
            "skipped": skipped, "flagged": flagged}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371000.0 * 2 * math.asin(math.sqrt(a))


def cameras_near(conn: sqlite3.Connection, lat: float, lon: float, radius_m: float,
                 camera_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Cameras within radius of a point, nearest first. Cameras without coordinates are excluded."""
    out = []
    for r in conn.execute("SELECT * FROM cw_cameras WHERE latitude IS NOT NULL AND longitude IS NOT NULL"):
        if camera_types and r["camera_type"] not in camera_types:
            continue
        d = haversine_m(lat, lon, r["latitude"], r["longitude"])
        if d <= radius_m:
            out.append({**dict(r), "distance_m": round(d, 1)})
    return sorted(out, key=lambda x: x["distance_m"])
