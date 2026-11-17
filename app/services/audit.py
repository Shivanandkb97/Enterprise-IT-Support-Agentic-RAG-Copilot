
import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()


def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)

    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )

    # Store uploaded document information for duplicate detection
    con.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            indexed_at TEXT NOT NULL
        )"""
    )

    con.commit()
    con.close()


def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)

    con.execute(
        """INSERT INTO query_audit
        (created_at, question, source_used, trace_json)
        VALUES (?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            source_used,
            json.dumps(trace),
        ),
    )

    con.commit()
    con.close()


# Check whether a document has already been indexed
def is_document_duplicate(file_hash: str) -> bool:
    con = sqlite3.connect(settings.audit_db_path)

    try:
        result = con.execute(
            "SELECT 1 FROM documents WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()

        return result is not None

    finally:
        con.close()


# Register a document after successful indexing
def register_document(file_name: str, file_hash: str) -> bool:
    con = sqlite3.connect(settings.audit_db_path)

    try:
        con.execute(
            """INSERT INTO documents (file_name, file_hash, indexed_at)
            VALUES (?, ?, ?)""",
            (
                file_name,
                file_hash,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        con.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        con.close()
