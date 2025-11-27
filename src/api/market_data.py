import requests
import logging
from src.core.config import Config

logger = logging.getLogger(__name__)

async def get_market_data(symbol: str, interval: str = "1min", limit: int = 100):
    """Get market data from TwelveData"""
    try:
        if not Config.TWELVEDATA_KEYS:
            logger.warning("No TwelveData keys configured")
            return get_sample_data()  # Fallback to sample data
        
        # Use first available key
        key = Config.TWELVEDATA_KEYS[0]
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": limit,
            "apikey": key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("values"):
            logger.info(f"✅ Market data received for {symbol}")
            return data
        else:
            logger.warning(f"❌ No data from TwelveData: {data.get('message')}")
            return get_sample_data()
            
    except Exception as e:
        logger.error(f"❌ Market data error: {e}")
        return get_sample_data()

def get_sample_data():
    """Return sample data when API fails"""
    return {
        "values": [
            {"datetime": "2024-01-01 10:00:00", "open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0855},
            {"datetime": "2024-01-01 10:01:00", "open": 1.0855, "high": 1.0865, "low": 1.0845, "close": 1.0860},
        ]
    }
