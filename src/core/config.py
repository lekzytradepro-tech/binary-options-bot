import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Core configuration for Binary Options AI Pro"""
    
    # ==================== BOT CONFIGURATION ====================
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/bot.db")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    PORT = int(os.getenv("PORT", "8000"))
    
    # ==================== ADMIN CONFIGURATION ====================
    ADMIN_IDS = []
    admin_env = os.getenv("ADMIN_IDS", "")
    if admin_env:
        try:
            ADMIN_IDS = [int(id.strip()) for id in admin_env.split(",") if id.strip()]
        except ValueError:
            logger.warning("Invalid ADMIN_IDS format")
    
    # ==================== API KEYS CONFIGURATION ====================
    
    # TwelveData API (PRIMARY - Your free keys)
    TWELVEDATA_KEYS = []
    td_keys_env = os.getenv("TWELVEDATA_KEYS", "")
    if td_keys_env:
        TWELVEDATA_KEYS = [key.strip() for key in td_keys_env.split(",") if key.strip()]
    
    # Fallback APIs (FREE alternatives)
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
    
    # ==================== FEATURE FLAGS ====================
    ENABLE_AI = os.getenv("ENABLE_AI", "true").lower() == "true"
    ENABLE_PAYMENTS = os.getenv("ENABLE_PAYMENTS", "false").lower() == "true"
    ENABLE_ADMIN = os.getenv("ENABLE_ADMIN", "true").lower() == "true"
    ENABLE_SIGNALS = os.getenv("ENABLE_SIGNALS", "true").lower() == "true"
    
    # ==================== TRADING CONFIGURATION ====================
    TRADING_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
        "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
        "XAU/USD", "XAG/USD", "BTC/USD", "ETH/USD", "US30/USD"
    ]
    
    SUPPORTED_EXPIRIES = ["1MIN", "5MIN", "15MIN", "1H", "4H"]
    
    # ==================== SUBSCRIPTION PLANS ====================
    PLANS = {
        "FREE_TRIAL": {
            "name": "🎯 Free Trial",
            "signals_per_day": 3,
            "strategies": ["TREND_SPOTTER", "ADAPTIVE_FILTER"],
            "assets": ["EUR/USD", "GBP/USD", "BTC/USD"],
            "expiries": ["5MIN", "15MIN"]
        },
        "BASIC": {
            "name": "⚡ Basic",
            "signals_per_day": 20,
            "strategies": ["TREND_SPOTTER", "ADAPTIVE_FILTER", "PATTERN_SNIPER"],
            "assets": "15+ Major Pairs",
            "expiries": ["1MIN", "5MIN", "15MIN", "1H"]
        },
        "PRO": {
            "name": "🚀 Pro", 
            "signals_per_day": "Unlimited",
            "strategies": "ALL",
            "assets": "ALL",
            "expiries": "ALL"
        }
    }
    
    # ==================== AI CONFIGURATION ====================
    AI_ENGINES = [
        "QuantumAIFusion", "AdaptiveMomentum", "TrendAnalysis", 
        "MeanReversion", "VolatilityAnalysis", "AIScalperEngine",
        "AIPulseEngine", "AIFlowMapEngine", "AISmartGridEngine",
        "QuantumReinforcement", "NeuralWavePattern", "LiquiditySweep",
        "PrecisionTiming", "DeepNeuralPattern", "QuantumEntanglement"
    ]
    
    # ==================== RISK MANAGEMENT ====================
    MAX_TRADE_SIZE = float(os.getenv("MAX_TRADE_SIZE", "1000"))
    DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "500"))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2%
    
    # ==================== API SETTINGS ====================
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "15"))
    API_RETRIES = int(os.getenv("API_RETRIES", "3"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    
    # ==================== SYSTEM SETTINGS ====================
    DATA_DIR = os.getenv("DATA_DIR", "data")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs")
    
    # ==================== VALIDATION METHODS ====================
    @classmethod
    def validate(cls):
        """Validate essential configuration"""
        errors = []
        
        # Check Telegram Token
        if not cls.TELEGRAM_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        
        # Check TwelveData Keys (primary API)
        if not cls.TWELVEDATA_KEYS:
            logger.warning("No TwelveData API keys configured. Using fallback APIs only.")
        
        # Check Database URL
        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL is required")
        
        if errors:
            error_msg = "Configuration errors:\n- " + "\n- ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuration validated successfully")
        logger.info(f"🤖 Bot Environment: {cls.ENVIRONMENT}")
        logger.info(f"🔑 TwelveData Keys: {len(cls.TWELVEDATA_KEYS)} configured")
        logger.info(f"👑 Admin Users: {len(cls.ADMIN_IDS)}")
        logger.info(f"📊 Trading Pairs: {len(cls.TRADING_PAIRS)}")
        logger.info(f"🤖 AI Engines: {len(cls.AI_ENGINES)}")
        
        return True
    
    @classmethod
    def get_api_status(cls):
        """Get API configuration status"""
        return {
            "twelvedata": {
                "configured": len(cls.TWELVEDATA_KEYS) > 0,
                "key_count": len(cls.TWELVEDATA_KEYS),
                "status": "✅ PRIMARY" if cls.TWELVEDATA_KEYS else "❌ NOT CONFIGURED"
            },
            "alphavantage": {
                "configured": bool(cls.ALPHAVANTAGE_API_KEY),
                "status": "✅ FALLBACK" if cls.ALPHAVANTAGE_API_KEY else "❌ NOT CONFIGURED"
            },
            "finnhub": {
                "configured": bool(cls.FINNHUB_API_KEY),
                "status": "✅ FALLBACK" if cls.FINNHUB_API_KEY else "❌ NOT CONFIGURED"
            },
            "polygon": {
                "configured": bool(cls.POLYGON_API_KEY),
                "status": "✅ FALLBACK" if cls.POLYGON_API_KEY else "❌ NOT CONFIGURED"
            }
        }

# Validate configuration on import
Config.validate()
