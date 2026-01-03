import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'memory.db')

def connect():
    """Connect to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables."""
    conn = connect()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            topic TEXT NOT NULL,
            url TEXT,
            title TEXT,
            notes TEXT NOT NULL,
            outcome TEXT DEFAULT 'unknown',
            tags TEXT DEFAULT ''
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT (datetime('now')),
            topic TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.7,
            source_episode_id INTEGER,
            source_url TEXT,
            FOREIGN KEY (source_episode_id) REFERENCES episodes (id)
        )
    ''')
    
    
    # 2026-01-02: Check for missing columns (session_id) and migrate if needed
    try:
        cursor.execute("ALTER TABLE episodes ADD COLUMN session_id TEXT")
        print("Schema Update: Added session_id to episodes table.")
    except sqlite3.OperationalError:
        pass # Column likely already exists
        
    try:
        cursor.execute("ALTER TABLE facts ADD COLUMN session_id TEXT")
        print("Schema Update: Added session_id to facts table.")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

def add_episode(topic, notes, url=None, title=None, outcome='unknown', tags='', session_id=None):
    """Add a new episode to the database."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO episodes (topic, notes, url, title, outcome, tags, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (topic, notes, url, title, outcome, tags, session_id))
    episode_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return episode_id

def get_episode(episode_id):
    """Retrieve an episode by ID."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM episodes WHERE id = ?', (episode_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def list_recent_episodes(limit=20):
    """List the most recent episodes."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_fact(topic, subject, predicate, object_, confidence=0.7, source_episode_id=None, source_url=None, session_id=None):
    """Add a new fact to the database."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO facts (topic, subject, predicate, object, confidence, source_episode_id, source_url, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (topic, subject, predicate, object_, confidence, source_episode_id, source_url, session_id))
    fact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fact_id

def get_fact(fact_id):
    """Retrieve a fact by ID."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM facts WHERE id = ?', (fact_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
    return dict(row) if row else None

def get_episodes_by_ids(ids: list[int]) -> list[dict]:
    """Retrieve episodes by a list of IDs, maintaining order."""
    if not ids:
        return []
    
    conn = connect()
    cursor = conn.cursor()
    
    # Use IN clause for efficiency
    placeholders = ','.join('?' for _ in ids)
    cursor.execute(f'SELECT * FROM episodes WHERE id IN ({placeholders})', ids)
    rows = cursor.fetchall()
    conn.close()
    
    # Create a lookup map
    lookup = {row['id']: dict(row) for row in rows}
    
    # Reassemble in requested order, skipping missing
    return [lookup[i] for i in ids if i in lookup]

def get_facts_by_ids(ids: list[int]) -> list[dict]:
    """Retrieve facts by a list of IDs, maintaining order."""
    if not ids:
        return []

    conn = connect()
    cursor = conn.cursor()
    
    placeholders = ','.join('?' for _ in ids)
    cursor.execute(f'SELECT * FROM facts WHERE id IN ({placeholders})', ids)
    rows = cursor.fetchall()
    conn.close()
    
    lookup = {row['id']: dict(row) for row in rows}
    
    return [lookup[i] for i in ids if i in lookup]

def get_episodes_by_topic(topic: str) -> list[dict]:
    """Retrieve all episodes for a given topic (case-insensitive)."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM episodes WHERE lower(topic) = lower(?) ORDER BY id DESC', (topic,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_latest_session_id(topic: str) -> str:
    """Retrieve the session_id of the most recent episode for a topic."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT session_id FROM episodes WHERE lower(topic) = lower(?) AND session_id IS NOT NULL ORDER BY id DESC LIMIT 1', (topic,))
    row = cursor.fetchone()
    conn.close()
    return row['session_id'] if row else None

def get_episodes_by_topic_and_session(topic: str, session_id: str) -> list[dict]:
    """Retrieve episodes for a specific topic and session."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM episodes WHERE lower(topic) = lower(?) AND session_id = ? ORDER BY id DESC', (topic, session_id))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_facts_by_topic_and_session(topic: str, session_id: str) -> list[dict]:
    """Retrieve facts for a specific topic and session."""
    conn = connect()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM facts WHERE lower(topic) = lower(?) AND session_id = ? ORDER BY id DESC', (topic, session_id))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

if __name__ == '__main__':
    # Initial setup when run directly
    init_db()
    print(f"Database initialized at {DB_PATH}")
