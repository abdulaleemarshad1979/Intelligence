"""PostgreSQL Database Connector & Watchlist Target Storage.

Provides persistent relational storage for suspects, reference photos,
biometric embeddings, and live surveillance match events in PostgreSQL.
"""

import os
import json
import time
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default PostgreSQL Connection URI:
# Can be overridden via DATABASE_URL or POSTGRES_URL environment variables
DEFAULT_POSTGRES_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("POSTGRES_URL")
    or "postgresql://postgres:postgres@localhost:5432/cctv_intelligence"
)


class PostgresWatchlistDB:
    """Manages PostgreSQL tables and operations for suspect photos and biometric vectors with SQLite fallback."""

    def __init__(self, connection_url: Optional[str] = None):
        self.connection_url = connection_url or DEFAULT_POSTGRES_URL
        self.is_connected = False
        self.psycopg2 = None
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.sqlite_path = os.path.join(base_dir, "data", "watchlist_sqlite.db")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self._init_sqlite_tables()
        self._init_connection()

    def _get_sqlite_conn(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite_tables(self):
        try:
            with self._get_sqlite_conn() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_suspects (
                    id TEXT PRIMARY KEY,
                    target_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    image_url TEXT,
                    image_base64 TEXT,
                    embedding TEXT,
                    threshold REAL DEFAULT 0.55,
                    quality_score REAL DEFAULT 0.0,
                    laplacian_var REAL DEFAULT 0.0,
                    notes TEXT DEFAULT '',
                    fir_no TEXT DEFAULT '',
                    police_station TEXT DEFAULT '',
                    total_matches INTEGER DEFAULT 0,
                    last_seen_camera TEXT,
                    last_seen_time REAL,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_captures (
                    alert_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    target_name TEXT,
                    camera_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    similarity_pct REAL NOT NULL,
                    full_frame_path TEXT,
                    face_crop_path TEXT,
                    body_crop_path TEXT,
                    full_frame_url TEXT,
                    face_crop_url TEXT,
                    raw_frame_hash TEXT,
                    face_crop_hash TEXT,
                    timestamp REAL DEFAULT (strftime('%s', 'now'))
                );
                """)
                conn.commit()
        except Exception as ex:
            logger.debug(f"SQLite fallback table init note: {ex}")

    def _init_connection(self):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.psycopg2 = psycopg2
            self.RealDictCursor = RealDictCursor

            conn = self.get_connection()
            if conn:
                self.is_connected = True
                conn.close()
                self.init_tables()
                logger.info("Successfully connected to PostgreSQL surveillance database.")
        except Exception as ex:
            self.is_connected = False
            logger.warning(f"PostgreSQL connection note (will retry on demand): {ex}")

    def get_connection(self):
        """Creates a new PostgreSQL connection with timeout."""
        if not self.psycopg2:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                self.psycopg2 = psycopg2
                self.RealDictCursor = RealDictCursor
            except ImportError:
                return None

        try:
            conn = self.psycopg2.connect(
                self.connection_url,
                connect_timeout=1
            )
            return conn
        except Exception as ex:
            logger.debug(f"Unable to connect to PostgreSQL: {ex}")
            return None

    def init_tables(self) -> bool:
        """Initializes tables for suspect photos, biometric embeddings, and audit trails."""
        conn = self.get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                # 1. Watchlist Suspects Table (Photo, Metadata, Embedding)
                cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_suspects (
                    id VARCHAR(64) PRIMARY KEY,
                    target_id VARCHAR(64) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    image_path TEXT,
                    image_url TEXT,
                    image_base64 TEXT,
                    embedding JSONB,
                    threshold FLOAT DEFAULT 0.55,
                    quality_score FLOAT DEFAULT 0.0,
                    laplacian_var FLOAT DEFAULT 0.0,
                    notes TEXT DEFAULT '',
                    fir_no VARCHAR(128) DEFAULT '',
                    police_station VARCHAR(128) DEFAULT '',
                    total_matches INT DEFAULT 0,
                    last_seen_camera VARCHAR(64),
                    last_seen_time TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # 2. Watchlist Captures Table (CCTV Snapshots & SHA-256 Hashes)
                cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_captures (
                    alert_id VARCHAR(64) PRIMARY KEY,
                    target_id VARCHAR(64) NOT NULL,
                    target_name VARCHAR(255),
                    camera_id VARCHAR(64) NOT NULL,
                    confidence FLOAT NOT NULL,
                    similarity_pct FLOAT NOT NULL,
                    full_frame_path TEXT,
                    face_crop_path TEXT,
                    body_crop_path TEXT,
                    full_frame_url TEXT,
                    face_crop_url TEXT,
                    raw_frame_hash VARCHAR(64),
                    face_crop_hash VARCHAR(64),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                # Index on target_id and name for fast lookups
                cur.execute("CREATE INDEX IF NOT EXISTS idx_suspect_target_id ON watchlist_suspects (target_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_suspect_name ON watchlist_suspects (name);")
                conn.commit()
                self.is_connected = True
                return True
        except Exception as ex:
            logger.error(f"Error creating PostgreSQL tables: {ex}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def save_suspect(self, target_data: Dict[str, Any]) -> bool:
        """Upsert a suspect record with photo and embedding into PostgreSQL and SQLite."""
        tid = target_data.get("target_id") or f"TGT-{int(time.time())}"
        name = target_data.get("name", "Unknown Suspect")
        emb = target_data.get("embedding", [])
        emb_json = json.dumps(emb if isinstance(emb, list) else list(emb))
        thresh = float(target_data.get("threshold", 0.55))
        img_path = target_data.get("reference_path", "")
        img_url = target_data.get("reference_url", "")
        img_b64 = target_data.get("image_base64", "")
        notes = target_data.get("notes", "")
        fir_no = target_data.get("fir_no", "")
        quality = target_data.get("quality", {})
        q_score = float(quality.get("quality_score", 0.0))
        lap_var = float(quality.get("laplacian_var", 0.0))
        now_ts = time.time()

        # 1. Always persist to local SQLite database
        try:
            with self._get_sqlite_conn() as s_conn:
                s_conn.execute("""
                INSERT INTO watchlist_suspects (
                    id, target_id, name, image_path, image_url, image_base64,
                    embedding, threshold, quality_score, laplacian_var, notes, fir_no, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (target_id) DO UPDATE SET
                    name = excluded.name,
                    image_path = excluded.image_path,
                    image_url = excluded.image_url,
                    image_base64 = excluded.image_base64,
                    embedding = excluded.embedding,
                    threshold = excluded.threshold,
                    quality_score = excluded.quality_score,
                    laplacian_var = excluded.laplacian_var,
                    notes = excluded.notes,
                    fir_no = excluded.fir_no,
                    updated_at = excluded.updated_at;
                """, (
                    tid, tid, name, img_path, img_url, img_b64,
                    emb_json, thresh, q_score, lap_var, notes, fir_no, now_ts
                ))
                s_conn.commit()
        except Exception as s_ex:
            logger.debug(f"SQLite save note: {s_ex}")

        # 2. Persist to PostgreSQL if available
        conn = self.get_connection()
        if not conn:
            logger.info(f"Saved suspect '{name}' ({tid}) to local SQLite database.")
            return True

        try:
            query = """
            INSERT INTO watchlist_suspects (
                id, target_id, name, image_path, image_url, image_base64,
                embedding, threshold, quality_score, laplacian_var, notes, fir_no, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            )
            ON CONFLICT (target_id) DO UPDATE SET
                name = EXCLUDED.name,
                image_path = EXCLUDED.image_path,
                image_url = EXCLUDED.image_url,
                image_base64 = EXCLUDED.image_base64,
                embedding = EXCLUDED.embedding,
                threshold = EXCLUDED.threshold,
                quality_score = EXCLUDED.quality_score,
                laplacian_var = EXCLUDED.laplacian_var,
                notes = EXCLUDED.notes,
                fir_no = EXCLUDED.fir_no,
                updated_at = CURRENT_TIMESTAMP;
            """
            with conn.cursor() as cur:
                cur.execute(query, (
                    tid, tid, name, img_path, img_url, img_b64,
                    emb_json, thresh, q_score, lap_var, notes, fir_no
                ))
            conn.commit()
            self.is_connected = True
            logger.info(f"Saved suspect '{name}' ({tid}) to PostgreSQL database.")
            return True
        except Exception as ex:
            logger.error(f"Failed to save suspect to PostgreSQL (saved to SQLite): {ex}")
            conn.rollback()
            return True
        finally:
            conn.close()

    def get_all_suspects(self) -> List[Dict[str, Any]]:
        """Retrieve all enrolled suspects from PostgreSQL (or SQLite fallback)."""
        results = []
        conn = self.get_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=self.RealDictCursor) as cur:
                    cur.execute("""
                    SELECT id, target_id, name, image_path, image_url, image_base64,
                           embedding, threshold, quality_score, laplacian_var, notes,
                           fir_no, total_matches, last_seen_camera, last_seen_time,
                           created_at, updated_at
                    FROM watchlist_suspects
                    ORDER BY created_at DESC;
                    """)
                    rows = cur.fetchall()
                    for r in rows:
                        item = dict(r)
                        if isinstance(item.get("embedding"), str):
                            try:
                                item["embedding"] = json.loads(item["embedding"])
                            except Exception:
                                pass
                        if item.get("created_at") and hasattr(item["created_at"], "timestamp"):
                            item["enrolled_at"] = item["created_at"].timestamp()
                        results.append(item)
                    if results:
                        return results
            except Exception as ex:
                logger.error(f"Failed to query suspects from PostgreSQL, trying SQLite: {ex}")
            finally:
                conn.close()

        # SQLite Fallback Retrieval
        try:
            with self._get_sqlite_conn() as s_conn:
                rows = s_conn.execute("""
                SELECT id, target_id, name, image_path, image_url, image_base64,
                       embedding, threshold, quality_score, laplacian_var, notes,
                       fir_no, total_matches, last_seen_camera, last_seen_time,
                       created_at, updated_at
                FROM watchlist_suspects
                ORDER BY created_at DESC;
                """).fetchall()
                for r in rows:
                    item = dict(r)
                    if isinstance(item.get("embedding"), str):
                        try:
                            item["embedding"] = json.loads(item["embedding"])
                        except Exception:
                            pass
                    item["enrolled_at"] = item.get("created_at", time.time())
                    results.append(item)
        except Exception as s_ex:
            logger.debug(f"SQLite query error: {s_ex}")

        return results

    def delete_suspect(self, target_id: str) -> bool:
        """Remove a suspect from PostgreSQL and SQLite."""
        try:
            with self._get_sqlite_conn() as s_conn:
                s_conn.execute("DELETE FROM watchlist_suspects WHERE target_id = ? OR id = ?;", (target_id, target_id))
                s_conn.commit()
        except Exception:
            pass

        conn = self.get_connection()
        if not conn:
            return True

        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM watchlist_suspects WHERE target_id = %s OR id = %s;", (target_id, target_id))
            conn.commit()
            return True
        except Exception as ex:
            logger.error(f"Failed to delete suspect from PostgreSQL: {ex}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def clear_all(self) -> bool:
        """Wipe all suspects in PostgreSQL and SQLite watchlist."""
        try:
            with self._get_sqlite_conn() as s_conn:
                s_conn.execute("DELETE FROM watchlist_suspects;")
                s_conn.commit()
        except Exception:
            pass

        conn = self.get_connection()
        if not conn:
            return True

        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM watchlist_suspects;")
            conn.commit()
            return True
        except Exception as ex:
            logger.error(f"Failed to clear PostgreSQL watchlist: {ex}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def save_capture(self, capture_event: Dict[str, Any]) -> bool:
        """Save a match capture event into PostgreSQL and SQLite."""
        try:
            with self._get_sqlite_conn() as s_conn:
                s_conn.execute("""
                INSERT OR IGNORE INTO watchlist_captures (
                    alert_id, target_id, target_name, camera_id, confidence,
                    similarity_pct, full_frame_path, face_crop_path, body_crop_path,
                    full_frame_url, face_crop_url, raw_frame_hash, face_crop_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    capture_event.get("alert_id"),
                    capture_event.get("target_id"),
                    capture_event.get("target_name"),
                    capture_event.get("camera_id"),
                    capture_event.get("confidence", 0.0),
                    capture_event.get("similarity_pct", 0.0),
                    capture_event.get("full_frame_path"),
                    capture_event.get("face_crop_path"),
                    capture_event.get("body_crop_path"),
                    capture_event.get("full_frame_url"),
                    capture_event.get("face_crop_url"),
                    capture_event.get("raw_frame_hash"),
                    capture_event.get("face_crop_hash")
                ))
                s_conn.commit()
        except Exception:
            pass

        conn = self.get_connection()
        if not conn:
            return True

        try:
            query = """
            INSERT INTO watchlist_captures (
                alert_id, target_id, target_name, camera_id, confidence,
                similarity_pct, full_frame_path, face_crop_path, body_crop_path,
                full_frame_url, face_crop_url, raw_frame_hash, face_crop_hash
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (alert_id) DO NOTHING;
            """
            with conn.cursor() as cur:
                cur.execute(query, (
                    capture_event.get("alert_id"),
                    capture_event.get("target_id"),
                    capture_event.get("target_name"),
                    capture_event.get("camera_id"),
                    capture_event.get("confidence", 0.0),
                    capture_event.get("similarity_pct", 0.0),
                    capture_event.get("full_frame_path"),
                    capture_event.get("face_crop_path"),
                    capture_event.get("body_crop_path"),
                    capture_event.get("full_frame_url"),
                    capture_event.get("face_crop_url"),
                    capture_event.get("raw_frame_hash"),
                    capture_event.get("face_crop_hash")
                ))
            conn.commit()
            return True
        except Exception as ex:
            logger.error(f"Failed to save capture to PostgreSQL: {ex}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_status(self) -> Dict[str, Any]:
        """Check PostgreSQL server status, latency, and counts."""
        t0 = time.perf_counter()
        conn = self.get_connection()
        if not conn:
            return {
                "connected": False,
                "database": "PostgreSQL (Disconnected - SQLite Fallback Active)",
                "error": "Connection refused or server offline",
                "total_records": 0
            }

        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                v_str = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM watchlist_suspects;")
                count = cur.fetchone()[0]
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "connected": True,
                "database": "PostgreSQL 16 (Connected)",
                "version": v_str.split("on")[0].strip(),
                "total_records": count,
                "ping_ms": latency_ms,
                "connection_url": self.connection_url.split("@")[-1]  # Hide credentials
            }
        except Exception as ex:
            return {
                "connected": False,
                "database": "PostgreSQL (Error)",
                "error": str(ex),
                "total_records": 0
            }
        finally:
            conn.close()


# Global singleton instance
postgres_db = PostgresWatchlistDB()
