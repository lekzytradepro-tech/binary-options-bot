import sqlite3
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path="data/bot.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        import os
        os.makedirs("data", exist_ok=True)
        
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    plan TEXT DEFAULT 'FREE_TRIAL',
                    signals_used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("✅ Database initialized with user management")
    
    @contextmanager 
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def add_user(self, telegram_id, username=None, first_name=None):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (telegram_id, username, first_name) VALUES (?, ?, ?)",
                (telegram_id, username, first_name)
            )

# Global instance
db = Database()
