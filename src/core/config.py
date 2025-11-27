import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    # ... existing code ...
    
    # BINARY OPTIONS SPECIFIC CONFIG
    BINARY_PAIRS = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
        "USD/CAD", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
        "XAU/USD", "XAG/USD", "BTC/USD", "ETH/USD", "US30/USD"
    ]
    
    # Binary Options Expiry Times (in minutes)
    BINARY_EXPIRIES = [1, 2, 5, 15, 30, 60]
    
    # Binary Options Types
    BINARY_TYPES = ["CALL/PUT", "TOUCH/NO_TOUCH", "BOUNDARY", "DIGITAL"]
    
    # Payout percentages based on volatility
    PAYOUT_RATES = {
        "LOW_VOLATILITY": {"CALL/PUT": 70, "TOUCH": 150, "BOUNDARY": 180},
        "MEDIUM_VOLATILITY": {"CALL/PUT": 75, "TOUCH": 200, "BOUNDARY": 250},
        "HIGH_VOLATILITY": {"CALL/PUT": 85, "TOUCH": 300, "BOUNDARY": 400}
    }
    
    # TwelveData API for real binary data
    TWELVEDATA_KEYS = []
    td_keys_env = os.getenv("TWELVEDATA_KEYS", "")
    if td_keys_env:
        TWELVEDATA_KEYS = [key.strip() for key in td_keys_env.split(",") if key.strip()]
    
    # AI Engines optimized for Binary Options
    BINARY_AI_ENGINES = [
        "BinaryTrendAI", "VolatilityAI", "ExpiryAI", "MomentumAI", 
        "PatternAI", "LiquidityAI", "NewsAI", "SessionAI"
    ]

Config.validate()
