import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for Binary Options AI Pro"""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    TELEGRAM_ADMIN_IDS = [int(x) for x in os.getenv("TELEGRAM_ADMIN_IDS", "").split(",") if x]
    
    # TwelveData API Configuration
    TWELVEDATA_KEYS = [key.strip() for key in os.getenv("TWELVEDATA_KEYS", "").split(",") if key.strip()]
    TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///binary_bot.db")
    
    # Binary Options Trading Configuration
    BINARY_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
        "USD/CAD", "NZD/USD", "EUR/GBP", "GBP/JPY", "EUR/JPY",
        "BTC/USD", "ETH/USD", "XAU/USD", "XAG/USD", "US30"
    ]
    
    # Binary Expiry Times (in minutes)
    BINARY_EXPIRIES = [1, 2, 5, 15, 30, 60]
    
    # AI Engines for Binary Options
    BINARY_AI_ENGINES = {
        "trend_analyzer": "Trend Analysis AI",
        "momentum_detector": "Momentum Detection AI", 
        "volatility_ai": "Volatility Analysis AI",
        "pattern_recognizer": "Pattern Recognition AI",
        "sentiment_analyzer": "Market Sentiment AI",
        "rsi_analyzer": "RSI Analysis AI",
        "breakout_detector": "Breakout Detection AI",
        "news_analyzer": "News Impact AI"
    }
    
    # Payout Rates by Asset Type (%)
    PAYOUT_RATES = {
        "forex": {"min": 75, "max": 85},
        "crypto": {"min": 80, "max": 90},
        "commodities": {"min": 70, "max": 80},
        "indices": {"min": 75, "max": 85}
    }
    
    # Trading Hours Configuration (UTC)
    TRADING_SESSIONS = {
        "asian": {"start": 22, "end": 6},      # 22:00 - 06:00 UTC
        "london": {"start": 7, "end": 16},     # 07:00 - 16:00 UTC  
        "new_york": {"start": 12, "end": 21},  # 12:00 - 21:00 UTC
        "overlap": {"start": 12, "end": 16}    # London/NY overlap
    }
    
    # Risk Management Configuration
    MAX_TRADES_PER_DAY = 10
    MIN_CONFIDENCE_THRESHOLD = 65  # Minimum AI confidence percentage
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
    
    # Signal Generation Configuration
    SIGNAL_VALIDITY_MINUTES = 2
    SIGNAL_REFRESH_SECONDS = 30
    
    # Technical Analysis Configuration
    TECHNICAL_INDICATORS = {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bollinger_period": 20,
        "bollinger_std": 2,
        "atr_period": 14,
        "sma_short": 10,
        "sma_medium": 20,
        "sma_long": 50
    }
    
    # Volatility Settings
    VOLATILITY_THRESHOLDS = {
        "low": 0.005,    # 0.5%
        "medium": 0.015, # 1.5%
        "high": 0.03     # 3.0%
    }
    
    # Asset-Specific Configuration
    ASSET_CONFIGS = {
        "EUR/USD": {
            "type": "forex",
            "volatility": "medium",
            "spread": 0.0001,
            "recommended_expiry": [5, 15, 30],
            "session": ["london", "new_york", "overlap"]
        },
        "GBP/USD": {
            "type": "forex", 
            "volatility": "medium",
            "spread": 0.0002,
            "recommended_expiry": [5, 15, 30],
            "session": ["london", "new_york", "overlap"]
        },
        "USD/JPY": {
            "type": "forex",
            "volatility": "medium",
            "spread": 0.0003,
            "recommended_expiry": [5, 15, 30],
            "session": ["asian", "london", "new_york"]
        },
        "BTC/USD": {
            "type": "crypto",
            "volatility": "high", 
            "spread": 0.001,
            "recommended_expiry": [2, 5, 15],
            "session": ["24/7"]
        },
        "ETH/USD": {
            "type": "crypto",
            "volatility": "high",
            "spread": 0.0015,
            "recommended_expiry": [2, 5, 15],
            "session": ["24/7"]
        },
        "XAU/USD": {
            "type": "commodity",
            "volatility": "medium",
            "spread": 0.05,
            "recommended_expiry": [15, 30, 60],
            "session": ["london", "new_york"]
        },
        "US30": {
            "type": "index",
            "volatility": "medium",
            "spread": 0.5,
            "recommended_expiry": [15, 30, 60],
            "session": ["new_york"]
        }
    }
    
    # AI Model Configuration
    AI_CONFIG = {
        "confidence_calibration": {
            "high_confidence": 80,
            "medium_confidence": 65,
            "low_confidence": 50
        },
        "signal_weights": {
            "trend_strength": 0.25,
            "momentum": 0.20,
            "volatility": 0.15,
            "pattern": 0.15,
            "support_resistance": 0.15,
            "market_sentiment": 0.10
        },
        "timeframe_analysis": ["1min", "5min", "15min", "1h"]
    }
    
    # User Account Configuration
    ACCOUNT_TIERS = {
        "free": {
            "signals_per_day": 3,
            "ai_engines": 4,
            "assets": 8,
            "refresh_rate": 300,  # 5 minutes
            "features": ["basic_signals", "standard_assets", "email_support"]
        },
        "premium": {
            "signals_per_day": 50,
            "ai_engines": 8,
            "assets": 15,
            "refresh_rate": 60,   # 1 minute
            "features": ["all_signals", "all_assets", "priority_support", "advanced_analytics"]
        },
        "vip": {
            "signals_per_day": 999,
            "ai_engines": 8,
            "assets": 15,
            "refresh_rate": 30,   # 30 seconds
            "features": ["unlimited_signals", "all_assets", "dedicated_support", "custom_strategies"]
        }
    }
    
    # Performance Monitoring
    PERFORMANCE = {
        "accuracy_target": 75,
        "max_drawdown": 5,
        "risk_reward_ratio": 2.0,
        "weekly_analysis": True
    }
    
    # Notification Settings
    NOTIFICATIONS = {
        "signal_alerts": True,
        "performance_updates": True,
        "market_news": True,
        "system_maintenance": True
    }
    
    # Cache Configuration
    CACHE_TIMEOUTS = {
        "market_data": 30,       # 30 seconds
        "technical_indicators": 60,  # 1 minute
        "ai_predictions": 120,   # 2 minutes
        "user_data": 300         # 5 minutes
    }
    
    # Logging Configuration
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "binary_bot.log"
    }

    @classmethod
    def validate(cls):
        """Validate configuration and set up binary options defaults"""
        logger.info("🔧 Validating Binary Options Configuration...")
        
        # Validate required API keys
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        
        if not cls.TWELVEDATA_KEYS:
            logger.warning("⚠️ No TwelveData API keys configured - using demo mode")
            cls.DEMO_MODE = True
        else:
            cls.DEMO_MODE = False
        
        # Validate binary pairs
        if not cls.BINARY_PAIRS:
            cls.BINARY_PAIRS = [
                "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
                "USD/CAD", "NZD/USD", "EUR/GBP", "GBP/JPY", "BTC/USD",
                "ETH/USD", "XAU/USD", "XAG/USD", "US30", "SPX"
            ]
            logger.info(f"📊 Using default binary pairs: {len(cls.BINARY_PAIRS)} assets")
        
        # Validate AI engines
        if not cls.BINARY_AI_ENGINES:
            cls.BINARY_AI_ENGINES = {
                "trend_analyzer": "Trend Analysis AI",
                "momentum_detector": "Momentum Detection AI", 
                "volatility_ai": "Volatility Analysis AI",
                "pattern_recognizer": "Pattern Recognition AI",
                "sentiment_analyzer": "Market Sentiment AI",
                "rsi_analyzer": "RSI Analysis AI",
                "breakout_detector": "Breakout Detection AI",
                "news_analyzer": "News Impact AI"
            }
            logger.info(f"🤖 Using default AI engines: {len(cls.BINARY_AI_ENGINES)} engines")
        
        # Validate expiries
        if not cls.BINARY_EXPIRIES:
            cls.BINARY_EXPIRIES = [1, 2, 5, 15, 30, 60]
            logger.info(f"⏰ Using default expiries: {cls.BINARY_EXPIRIES} minutes")
        
        # Validate payout rates
        if not cls.PAYOUT_RATES:
            cls.PAYOUT_RATES = {
                "forex": {"min": 75, "max": 85},
                "crypto": {"min": 80, "max": 90},
                "commodities": {"min": 70, "max": 80},
                "indices": {"min": 75, "max": 85}
            }
            logger.info("💰 Using default payout rates")
        
        # Validate asset configurations
        for asset in cls.BINARY_PAIRS:
            if asset not in cls.ASSET_CONFIGS:
                # Auto-configure missing assets
                if '/' in asset and 'XAU' not in asset and 'XAG' not in asset:
                    asset_type = "forex"
                elif 'BTC' in asset or 'ETH' in asset:
                    asset_type = "crypto"
                elif 'XAU' in asset or 'XAG' in asset:
                    asset_type = "commodity"
                else:
                    asset_type = "index"
                
                cls.ASSET_CONFIGS[asset] = {
                    "type": asset_type,
                    "volatility": "medium",
                    "spread": 0.0002,
                    "recommended_expiry": [5, 15, 30],
                    "session": ["london", "new_york"]
                }
                logger.info(f"⚙️ Auto-configured asset: {asset} as {asset_type}")
        
        logger.info("✅ Binary Options Configuration Validated Successfully")
        logger.info(f"🎯 Trading Assets: {len(cls.BINARY_PAIRS)}")
        logger.info(f"🤖 AI Engines: {len(cls.BINARY_AI_ENGINES)}")
        logger.info(f"⏰ Expiry Times: {cls.BINARY_EXPIRIES}")
        logger.info(f"💼 Demo Mode: {cls.DEMO_MODE}")

    @classmethod
    def get_asset_type(cls, asset: str) -> str:
        """Get asset type (forex, crypto, commodity, index)"""
        return cls.ASSET_CONFIGS.get(asset, {}).get("type", "forex")
    
    @classmethod
    def get_recommended_expiry(cls, asset: str) -> List[int]:
        """Get recommended expiry times for an asset"""
        return cls.ASSET_CONFIGS.get(asset, {}).get("recommended_expiry", [5, 15, 30])
    
    @classmethod
    def get_payout_rate(cls, asset: str, volatility: float) -> float:
        """Calculate payout rate based on asset type and volatility"""
        asset_type = cls.get_asset_type(asset)
        base_rates = cls.PAYOUT_RATES.get(asset_type, {"min": 75, "max": 85})
        
        # Adjust payout based on volatility (higher volatility = higher payout)
        if volatility > cls.VOLATILITY_THRESHOLDS["high"]:
            return base_rates["max"]
        elif volatility > cls.VOLATILITY_THRESHOLDS["medium"]:
            return (base_rates["min"] + base_rates["max"]) / 2
        else:
            return base_rates["min"]
    
    @classmethod
    def get_trading_sessions(cls, asset: str) -> List[str]:
        """Get active trading sessions for an asset"""
        return cls.ASSET_CONFIGS.get(asset, {}).get("session", ["london", "new_york"])
    
    @classmethod
    def get_ai_engines_for_asset(cls, asset: str) -> List[str]:
        """Get relevant AI engines for specific asset type"""
        asset_type = cls.get_asset_type(asset)
        
        engine_priority = {
            "forex": ["trend_analyzer", "momentum_detector", "sentiment_analyzer", "pattern_recognizer"],
            "crypto": ["volatility_ai", "trend_analyzer", "momentum_detector", "breakout_detector"],
            "commodity": ["trend_analyzer", "sentiment_analyzer", "volatility_ai", "news_analyzer"],
            "index": ["trend_analyzer", "sentiment_analyzer", "pattern_recognizer", "news_analyzer"]
        }
        
        return engine_priority.get(asset_type, list(cls.BINARY_AI_ENGINES.keys()))

# Validate configuration on import
try:
    Config.validate()
except Exception as e:
    logger.error(f"❌ Configuration validation failed: {e}")
    raise
