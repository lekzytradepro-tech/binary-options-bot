# 📁 admin/dashboard.py
import logging
from accounts.tier_manager import TierManager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdminDashboard:
    def __init__(self):
        self.tier_manager = TierManager()
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        try:
            with self.tier_manager._get_connection() as conn:
                # Total users
                total_users = conn.execute(
                    "SELECT COUNT(*) as count FROM account_users"
                ).fetchone()['count']
                
                # Today's signals
                today = datetime.now().date()
                today_signals = conn.execute(
                    "SELECT COUNT(*) as count FROM account_signal_usage WHERE date(created_at) = ?",
                    (today,)
                ).fetchone()['count']
                
                # Active users today
                active_users = conn.execute(
                    "SELECT COUNT(DISTINCT telegram_id) as count FROM account_signal_usage WHERE date(created_at) = ?",
                    (today,)
                ).fetchone()['count']
                
                # Tier distribution
                tiers = conn.execute(
                    "SELECT tier, COUNT(*) as count FROM account_users GROUP BY tier"
                ).fetchall()
                
                # Revenue calculation (for paid tiers)
                revenue = conn.execute(
                    "SELECT SUM(amount) as total FROM account_payments WHERE status = 'completed'"
                ).fetchone()['total'] or 0
                
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "users": {
                    "total": total_users,
                    "active_today": active_users
                },
                "signals": {
                    "today": today_signals,
                    "total": self._get_total_signals()
                },
                "tiers": {tier['tier']: tier['count'] for tier in tiers},
                "revenue": {
                    "total": revenue,
                    "currency": "USD"
                }
            }
            
        except Exception as e:
            logger.error(f"Admin stats error: {e}")
            return {"error": str(e)}
    
    def _get_total_signals(self):
        """Get total signals count"""
        try:
            with self.tier_manager._get_connection() as conn:
                result = conn.execute(
                    "SELECT COUNT(*) as count FROM account_signal_usage"
                ).fetchone()
                return result['count'] if result else 0
        except:
            return 0
    
    def upgrade_user(self, telegram_id, new_tier, duration_days=30):
        """Manually upgrade user (for admin use)"""
        return self.tier_manager.upgrade_user_tier(telegram_id, new_tier, duration_days)
    
    def get_user_details(self, telegram_id):
        """Get detailed user information"""
        try:
            with self.tier_manager._get_connection() as conn:
                user = conn.execute(
                    "SELECT * FROM account_users WHERE telegram_id = ?", 
                    (telegram_id,)
                ).fetchone()
                
                if user:
                    signals = conn.execute(
                        "SELECT COUNT(*) as count FROM account_signal_usage WHERE telegram_id = ?",
                        (telegram_id,)
                    ).fetchone()
                    
                    return {
                        "telegram_id": user['telegram_id'],
                        "username": user['username'],
                        "tier": user['tier'],
                        "tier_expires": user['tier_expires_at'],
                        "signals_used_today": user['signals_used_today'],
                        "total_signals": signals['count'] if signals else 0,
                        "created_at": user['created_at']
                    }
                return None
                
        except Exception as e:
            logger.error(f"Get user details error: {e}")
            return None
