# 📁 src/accounts/tier_manager.py
from datetime import datetime, timedelta
from .models import User
import logging

logger = logging.getLogger(__name__)

class TierManager:
    """Manages user tiers and feature access"""
    
    # Feature matrix for each tier
    TIER_FEATURES = {
        'free_trial': {
            'daily_signal_limit': 10,
            'assets': ['EUR/USD', 'GBP/USD', 'BTC/USD'],
            'ai_engines': ['binary_trend_ai'],
            'expiry_times': ['1min', '5min'],
            'refresh_rate_minutes': 30,
            'analytics': False,
            'api_access': False,
            'premium_support': False
        },
        'basic': {
            'daily_signal_limit': 50,
            'assets': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'ETH/USD', 'XAU/USD'],
            'ai_engines': ['binary_trend_ai', 'volatility_ai'],
            'expiry_times': ['1min', '2min', '5min', '15min'],
            'refresh_rate_minutes': 15,
            'analytics': True,
            'api_access': False,
            'premium_support': False
        },
        'pro': {
            'daily_signal_limit': 9999,  # Essentially unlimited
            'assets': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 
                      'BTC/USD', 'ETH/USD', 'XAU/USD', 'US30'],
            'ai_engines': ['binary_trend_ai', 'volatility_ai', 'expiry_ai'],
            'expiry_times': ['1min', '2min', '5min', '15min', '30min', '1hour'],
            'refresh_rate_minutes': 5,
            'analytics': True,
            'api_access': True,
            'premium_support': True
        },
        'enterprise': {
            'daily_signal_limit': 9999,
            'assets': 'ALL',
            'ai_engines': 'ALL',
            'expiry_times': 'ALL',
            'refresh_rate_minutes': 1,
            'analytics': True,
            'api_access': True,
            'premium_support': True,
            'white_label': True,
            'custom_ai': True
        }
    }
    
    def __init__(self, db_session):
        self.db = db_session
    
    def get_user_tier(self, telegram_id: str) -> dict:
        """Get user's current tier and features"""
        user = self.db.query(User).filter_by(telegram_id=telegram_id).first()
        
        if not user:
            return self.TIER_FEATURES['free_trial']
        
        # Check if trial expired
        if user.tier == 'free_trial' and user.tier_expiration < datetime.utcnow():
            user.tier = 'free'  # Downgrade to free after trial
            self.db.commit()
        
        tier = user.tier if user.tier in self.TIER_FEATURES else 'free_trial'
        return self.TIER_FEATURES[tier]
    
    def can_generate_signal(self, telegram_id: str) -> tuple:
        """Check if user can generate a signal today"""
        user = self.db.query(User).filter_by(telegram_id=telegram_id).first()
        
        if not user:
            return False, "User not found"
        
        features = self.get_user_tier(telegram_id)
        
        # Reset daily counter if new day
        today = datetime.utcnow().date()
        if not user.last_signal_date or user.last_signal_date.date() != today:
            user.signals_used_today = 0
            user.last_signal_date = datetime.utcnow()
            self.db.commit()
        
        # Check daily limit
        if user.signals_used_today >= features['daily_signal_limit']:
            return False, f"Daily limit reached ({features['daily_signal_limit']} signals)"
        
        return True, "Can generate signal"
    
    def record_signal_usage(self, telegram_id: str):
        """Record that user generated a signal"""
        user = self.db.query(User).filter_by(telegram_id=telegram_id).first()
        if user:
            user.signals_used_today += 1
            user.signals_used_total += 1
            user.last_signal_date = datetime.utcnow()
            self.db.commit()
    
    def upgrade_user_tier(self, telegram_id: str, new_tier: str, duration_days: int = 30):
        """Upgrade user to new tier"""
        user = self.db.query(User).filter_by(telegram_id=telegram_id).first()
        if user:
            user.tier = new_tier
            user.tier_expiration = datetime.utcnow() + timedelta(days=duration_days)
            user.signals_used_today = 0  # Reset counter
            self.db.commit()
            logger.info(f"User {telegram_id} upgraded to {new_tier}")
            return True
        return False
