# 📁 src/accounts/models.py
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(String(50), unique=True, nullable=False)
    username = Column(String(100))
    first_name = Column(String(100))
    tier = Column(String(20), default='free_trial')  # free_trial, basic, pro, enterprise
    tier_expiration = Column(DateTime)
    signals_used_today = Column(Integer, default=0)
    signals_used_total = Column(Integer, default=0)
    last_signal_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    payment_method = Column(String(20))  # stripe, crypto, manual
    subscription_id = Column(String(100))
    
    # Free trial tracking
    trial_started = Column(DateTime)
    trial_ended = Column(DateTime)
    
    def __init__(self, telegram_id, username=None, first_name=None):
        self.telegram_id = telegram_id
        self.username = username
        self.first_name = first_name
        self.trial_started = datetime.utcnow()
        self.tier_expiration = datetime.utcnow() + timedelta(days=14)  # 14-day trial

class UserSession(Base):
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50))  # signal_generated, payment_success, etc.
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Payment(Base):
    __tablename__ = 'payments'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    amount = Column(Float)
    currency = Column(String(10), default='USD')
    tier = Column(String(20))
    payment_method = Column(String(20))
    payment_id = Column(String(100))  # Stripe/Crypto transaction ID
    status = Column(String(20))  # pending, completed, failed, refunded
    created_at = Column(DateTime, default=datetime.utcnow)
