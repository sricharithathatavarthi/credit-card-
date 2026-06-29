import sqlite3
import tempfile
import os
from datetime import datetime

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS thresholds (
    month_key TEXT PRIMARY KEY,
    threshold_pct REAL
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL NOT NULL,
    note TEXT,
    timestamp TEXT NOT NULL
);
"""


def test_transaction_crud():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.executescript(SCHEMA_SQL)
        conn.commit()
        # insert
        cur.execute("INSERT INTO transactions(amount, note, timestamp) VALUES(?,?,?)", (12.5, 't1', datetime.utcnow().isoformat()))
        conn.commit()
        cur.execute("SELECT id, amount FROM transactions")
        rows = cur.fetchall()
        assert len(rows) == 1
        tx_id = rows[0][0]
        # update
        cur.execute("UPDATE transactions SET amount = ? WHERE id = ?", (20.0, tx_id))
        conn.commit()
        cur.execute("SELECT amount FROM transactions WHERE id = ?", (tx_id,))
        updated = cur.fetchone()[0]
        assert float(updated) == 20.0
        # delete
        cur.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM transactions")
        assert cur.fetchone()[0] == 0
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
