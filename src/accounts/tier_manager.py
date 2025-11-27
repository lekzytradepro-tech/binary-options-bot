# 📁 accounts/tier_manager.py
import logging
import sqlite3
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TierManager:
    """COMPLETELY SEPARATE tier management - doesn't touch your existing files"""
    
    def __init__(self, db_path="data/accounts.db"):
        self.db_path = db_path
        self._init_database()
        
        # Tier configuration - SEPARATE from your existing bot
        self.tiers = {
            'free_trial': {
                'name': '14-Day Free Trial',
                'signals_daily': 10,
                'assets': ['EUR/USD', 'GBP/USD', 'BTC/USD', 'XAU/USD', 'US30'],
                'expiries': ['1', '5', '15'],
                'price': 0,
                'duration_days': 14,
                'color': '🆓'
            },
            'basic': {
                'name': 'Basic Plan', 
                'signals_daily': 50,
                'assets': 'all',
                'expiries': ['1', '2', '5', '15', '30'],
                'price': 19,
                'duration_days': 30,
                'color': '💚'
            },
            'pro': {
                'name': 'Pro Trader',
                'signals_daily': 9999,  # Unlimited
                'assets': 'all',
                'expiries': 'all',
                'price': 49, 
                'duration_days': 30,
                'color': '💎'
            },
            'enterprise': {
                'name': 'Enterprise',
                'signals_daily': 9999,
                'assets': 'all', 
                'expiries': 'all',
                'price': 149,
                'duration_days': 30,
                'color': '👑'
            }
        }
    
    def _init_database(self):
        """Initialize SEPARATE database for account management"""
        import os
        os.makedirs("data", exist_ok=True)
        
        with self._get_connection() as conn:
            # Users table for account management
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id TEXT UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    tier TEXT DEFAULT 'free_trial',
                    tier_expires_at DATETIME,
                    signals_used_today INTEGER DEFAULT 0,
                    last_signal_date DATE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Payments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    payment_method TEXT,
                    payment_id TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Signal usage tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_signal_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("✅ Account management database initialized")
    
    @contextmanager
    def _get_connection(self):
        """Get connection to SEPARATE account database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Account DB error: {e}")
            raise
        finally:
            conn.close()
    
    def ensure_user_exists(self, telegram_id, username=None, first_name=None):
        """Ensure user exists in account management system"""
        try:
            with self._get_connection() as conn:
                user = conn.execute(
                    "SELECT * FROM account_users WHERE telegram_id = ?", 
                    (telegram_id,)
                ).fetchone()
                
                if not user:
                    # New user - give them free trial
                    tier_expires = datetime.now() + timedelta(days=14)
                    conn.execute("""
                        INSERT INTO account_users 
                        (telegram_id, username, first_name, tier, tier_expires_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (telegram_id, username, first_name, 'free_trial', tier_expires))
                    logger.info(f"🎯 New account user: {telegram_id}")
                    return True
                return True
                
        except Exception as e:
            logger.error(f"Ensure user error: {e}")
            return False
    
    def get_user_tier(self, telegram_id):
        """Get user's current tier with expiration check"""
        try:
            with self._get_connection() as conn:
                user = conn.execute(
                    "SELECT tier, tier_expires_at FROM account_users WHERE telegram_id = ?",
                    (telegram_id,)
                ).fetchone()
                
                if not user:
                    return 'free_trial'
                
                # Check if trial expired
                if user['tier'] == 'free_trial' and datetime.now() > datetime.fromisoformat(user['tier_expires_at']):
                    # Trial expired - still allow basic access but with limits
                    return 'free_trial_expired'
                
                return user['tier']
                
        except Exception as e:
            logger.error(f"Get user tier error: {e}")
            return 'free_trial'
    
    def can_generate_signal(self, telegram_id):
        """Check if user can generate a signal today"""
        try:
            tier = self.get_user_tier(telegram_id)
            tier_info = self.tiers.get(tier, self.tiers['free_trial'])
            
            # Free trial expired - very limited access
            if tier == 'free_trial_expired':
                return False, "Free trial expired. Upgrade to continue trading."
            
            with self._get_connection() as conn:
                # Reset daily counter if new day
                today = datetime.now().date()
                user = conn.execute(
                    "SELECT signals_used_today, last_signal_date FROM account_users WHERE telegram_id = ?",
                    (telegram_id,)
                ).fetchone()
                
                if user and user['last_signal_date']:
                    last_date = datetime.fromisoformat(user['last_signal_date']).date()
                    if last_date != today:
                        # New day - reset counter
                        conn.execute(
                            "UPDATE account_users SET signals_used_today = 0, last_signal_date = ? WHERE telegram_id = ?",
                            (datetime.now(), telegram_id)
                        )
                        signals_used = 0
                    else:
                        signals_used = user['signals_used_today'] or 0
                else:
                    signals_used = 0
                
                # Check daily limit
                if signals_used >= tier_info['signals_daily']:
                    return False, f"Daily limit reached ({tier_info['signals_daily']} signals). Upgrade for more."
                
                return True, f"Signals today: {signals_used}/{tier_info['signals_daily']}"
                
        except Exception as e:
            logger.error(f"Can generate signal error: {e}")
            return True, "Error checking limits"  # Fallback to allow signals
    
    def record_signal_usage(self, telegram_id, asset, expiry):
        """Record that user generated a signal"""
        try:
            with self._get_connection() as conn:
                # Update signals used today
                conn.execute("""
                    UPDATE account_users 
                    SET signals_used_today = signals_used_today + 1,
                        last_signal_date = ?
                    WHERE telegram_id = ?
                """, (datetime.now(), telegram_id))
                
                # Record signal details
                conn.execute("""
                    INSERT INTO account_signal_usage (telegram_id, asset, expiry)
                    VALUES (?, ?, ?)
                """, (telegram_id, asset, expiry))
                
        except Exception as e:
            logger.error(f"Record signal usage error: {e}")
    
    def upgrade_user_tier(self, telegram_id, new_tier, duration_days=30):
        """Upgrade user to new tier"""
        try:
            tier_info = self.tiers.get(new_tier)
            if not tier_info:
                return False, "Invalid tier"
            
            tier_expires = datetime.now() + timedelta(days=duration_days)
            
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE account_users 
                    SET tier = ?, tier_expires_at = ?, signals_used_today = 0
                    WHERE telegram_id = ?
                """, (new_tier, tier_expires, telegram_id))
                
                logger.info(f"✅ User {telegram_id} upgraded to {new_tier}")
                return True, f"Upgraded to {tier_info['name']}"
                
        except Exception as e:
            logger.error(f"Upgrade user error: {e}")
            return False, str(e)
    
    def get_user_stats(self, telegram_id):
        """Get user statistics and tier information"""
        try:
            tier = self.get_user_tier(telegram_id)
            tier_info = self.tiers.get(tier, self.tiers['free_trial'])
            
            with self._get_connection() as conn:
                # Get today's usage
                today = datetime.now().date()
                signals_today = conn.execute(
                    "SELECT signals_used_today FROM account_users WHERE telegram_id = ?",
                    (telegram_id,)
                ).fetchone()
                
                # Total signals
                total_signals = conn.execute(
                    "SELECT COUNT(*) as count FROM account_signal_usage WHERE telegram_id = ?",
                    (telegram_id,)
                ).fetchone()
                
                signals_used = signals_today['signals_used_today'] if signals_today else 0
                total_count = total_signals['count'] if total_signals else 0
                
            return {
                'tier': tier,
                'tier_name': tier_info['name'],
                'signals_used_today': signals_used,
                'signals_daily_limit': tier_info['signals_daily'],
                'total_signals': total_count,
                'assets_available': tier_info['assets'],
                'color': tier_info['color'],
                'price': tier_info['price']
            }
            
        except Exception as e:
            logger.error(f"Get user stats error: {e}")
            return {
                'tier': 'free_trial',
                'tier_name': 'Free Trial',
                'signals_used_today': 0,
                'signals_daily_limit': 10,
                'total_signals': 0,
                'assets_available': ['EUR/USD', 'GBP/USD', 'BTC/USD'],
                'color': '🆓',
                'price': 0
                    }
