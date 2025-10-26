# src/database/metadata_db.py
import sqlite3
import os
from typing import Optional, Dict, Tuple
from datetime import datetime


class MetadataDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(
            path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, check_same_thread=False)
        self._ensure()

    def _ensure(self):
        cur = self.conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            name TEXT,
            size INTEGER,
            created_at TEXT,
            modified_at TEXT,
            accessed_at TEXT,
            active INTEGER DEFAULT 1, -- Default to active
            -- NEW Behavioral Columns --
            access_count INTEGER DEFAULT 0,
            total_time_spent_hrs REAL DEFAULT 0.0,
            -- Keep extra_json if needed for future flexibility --
            extra_json TEXT DEFAULT '{}' 
        )
        ''')
        # Check and add new columns if they don't exist (for upgrades)
        self._add_column_if_not_exists(
            'files', 'access_count', 'INTEGER DEFAULT 0')
        self._add_column_if_not_exists(
            'files', 'total_time_spent_hrs', 'REAL DEFAULT 0.0')
        self.conn.commit()

    def _add_column_if_not_exists(self, table_name, column_name, column_type):
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cur.fetchall()]
        if column_name not in columns:
            print(f"Adding column '{column_name}' to table '{table_name}'...")
            cur.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            self.conn.commit()

    def upsert(self, meta: Dict):
        """Upserts file metadata, including behavioral data if provided."""
        cur = self.conn.cursor()
        # Ensure default values if behavioral data is missing in meta dict
        access_count = meta.get('access_count', 0)
        time_spent = meta.get('total_time_spent_hrs', 0.0)

        cur.execute('''
        INSERT INTO files (
            path, name, size, created_at, modified_at, accessed_at, 
            active, access_count, total_time_spent_hrs, extra_json
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
            name=excluded.name,
            size=excluded.size,
            created_at=excluded.created_at,
            modified_at=excluded.modified_at,
            accessed_at=excluded.accessed_at,
            active=excluded.active,
            access_count=excluded.access_count,
            total_time_spent_hrs=excluded.total_time_spent_hrs,
            extra_json=excluded.extra_json
        ''', (
            meta.get('path'), meta.get('name'), meta.get('size'),
            meta.get('created_at'), meta.get(
                'modified_at'), meta.get('accessed_at'),
            1,  # Mark as active on upsert
            access_count,
            time_spent,
            meta.get('extra_json', '{}')
        ))
        self.conn.commit()

    def mark_deleted(self, path: str):
        cur = self.conn.cursor()
        cur.execute('UPDATE files SET active=0 WHERE path=?', (path,))
        self.conn.commit()

    def fetch_all_active(self):
        """Fetches path and name for all active files."""
        cur = self.conn.cursor()
        # Fetch only path and name, which is likely enough for the simulator
        cur.execute('SELECT path, name FROM files WHERE active=1')
        return cur.fetchall()

    def get(self, path: str) -> Optional[Dict]:
        """Fetches all data for a specific file path."""
        cur = self.conn.cursor()
        cur.execute('SELECT * FROM files WHERE path=?', (path,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def get_metadata_for_retrieval(self, path: str) -> Optional[Tuple[str, str, float, int]]:
        """Fetches specific metadata needed for filtering/re-ranking."""
        cur = self.conn.cursor()
        cur.execute('''
            SELECT modified_at, accessed_at, total_time_spent_hrs, access_count 
            FROM files 
            WHERE path=? AND active=1
        ''', (path,))
        row = cur.fetchone()
        return row if row else None

    def get_modified_time(self, path: str) -> Optional[str]:
        """Gets only the stored modified time (ISO format) for an active file."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT modified_at FROM files WHERE path=? AND active=1', (path,))
        row = cur.fetchone()
        return row[0] if row else None

    # --- NEW Methods for Behavioral Data ---
    def increment_access(self, path: str, time_increment_hrs: float = 0.0):
        """Increments access count and optionally adds time spent for a file."""
        cur = self.conn.cursor()
        cur.execute("""
            UPDATE files 
            SET access_count = access_count + 1,
                total_time_spent_hrs = total_time_spent_hrs + ?,
                accessed_at = ? -- Update last accessed time
            WHERE path = ? AND active = 1
        """, (time_increment_hrs, datetime.now().isoformat(), path))

        # Check if the update affected any row
        updated_rows = cur.rowcount
        self.conn.commit()

        if updated_rows == 0:
            print(
                f"Warning: Tried to increment access for non-existent or inactive file: {path}")
            # Optionally, you could try to upsert basic metadata here if the file exists
            # but wasn't in the DB, though the watcher should handle this.

    def get_behavioral_data(self, path: str) -> Optional[Tuple[int, float]]:
        """Gets access_count and total_time_spent_hrs for a file."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT access_count, total_time_spent_hrs FROM files WHERE path=? AND active=1', (path,))
        row = cur.fetchone()
        return row if row else (0, 0.0)  # Return defaults if not found
