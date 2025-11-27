# 📁 utils/account_helpers.py
import logging
from accounts.tier_manager import TierManager

logger = logging.getLogger(__name__)

# Global tier manager instance
tier_manager = TierManager()

def initialize_user(telegram_id, username=None, first_name=None):
    """Initialize user in account system - call this from your existing bot"""
    return tier_manager.ensure_user_exists(telegram_id, username, first_name)

def check_signal_access(telegram_id):
    """Check if user can generate signal - call before generating signals"""
    return tier_manager.can_generate_signal(telegram_id)

def record_signal(telegram_id, asset, expiry):
    """Record signal usage - call after generating signal"""
    tier_manager.record_signal_usage(telegram_id, asset, expiry)

def get_user_account_info(telegram_id):
    """Get user account information for display"""
    return tier_manager.get_user_stats(telegram_id)

def get_available_assets(telegram_id):
    """Get available assets for user's tier"""
    stats = get_user_account_info(telegram_id)
    assets = stats['assets_available']
    
    if assets == 'all':
        # All 15 assets from your existing bot
        return [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
            "USD/CAD", "NZD/USD", "EUR/GBP", "GBP/JPY", "EUR/JPY",
            "BTC/USD", "ETH/USD", "XAU/USD", "XAG/USD", "US30"
        ]
    else:
        return assets
