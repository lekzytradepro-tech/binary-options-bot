import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_tables()
    
    def _ensure_data_dir(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_tables(self):
        with self.get_connection() as conn:
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Database tables initialized")
    
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
    
    def add_user(self, telegram_id: int, username: str = None, first_name: str = None):
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (telegram_id, username, first_name) VALUES (?, ?, ?)",
                (telegram_id, username, first_name)
            )
    
    def get_user(self, telegram_id: int):
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE telegram_id = ?",
                (telegram_id,)
            )
            return cursor.fetchone()

# Global instance
db = Database()
