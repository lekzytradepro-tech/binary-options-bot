import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Complete configuration with all AI engines and features"""
    
    # Bot Configuration
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/bot.db")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # AI Engines (All 15)
    AI_ENGINES = [
        "QuantumAIFusion", "AdaptiveMomentum", "TrendAnalysis", 
        "MeanReversion", "VolatilityAnalysis", "AIScalperEngine",
        "AIPulseEngine", "AIFlowMapEngine", "AISmartGridEngine", 
        "QuantumReinforcement", "NeuralWavePattern", "LiquiditySweep",
        "PrecisionTiming", "DeepNeuralPattern", "QuantumEntanglement"
    ]
    
    # Trading Strategies (All 7)
    TRADING_STRATEGIES = [
        "TREND_SPOTTER", "ADAPTIVE_FILTER", "PATTERN_SNIPER",
        "VOLUME_SPIKE", "SMARTTREND_PREDICTOR", "AI_SCALPER", 
        "QUANTUM_PULSE"
    ]
    
    # Trading Configuration
    TRADING_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
        "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
        "XAU/USD", "XAG/USD", "BTC/USD", "ETH/USD", "US30/USD"
    ]
    
    SUPPORTED_EXPIRIES = ["1MIN", "5MIN", "15MIN", "1H", "4H"]
    
    # Subscription Plans
    PLANS = {
        "FREE_TRIAL": {
            "name": "🎯 Free Trial",
            "signals_per_day": 3,
            "strategies": ["TREND_SPOTTER", "ADAPTIVE_FILTER"],
            "ai_engines": 4
        },
        "BASIC": {
            "name": "⚡ Basic", 
            "signals_per_day": 20,
            "strategies": "ALL",
            "ai_engines": 8
        },
        "PRO": {
            "name": "🚀 Pro",
            "signals_per_day": "Unlimited", 
            "strategies": "ALL",
            "ai_engines": "ALL 15"
        }
    }
    
    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")
        logger.info(f"✅ Config loaded: {len(cls.AI_ENGINES)} AI engines, {len(cls.TRADING_STRATEGIES)} strategies")
        return True

Config.validate()
