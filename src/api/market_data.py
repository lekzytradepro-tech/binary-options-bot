import requests
import logging
from typing import List, Dict, Any, Optional
from src.core.config import Config

logger = logging.getLogger(__name__)

class MarketDataAPI:
    def __init__(self):
        self.twelvedata_keys = Config.TWELVEDATA_KEYS
        self.current_key_index = 0
        self.fallback_apis = [
            self._yahoo_finance,
            self._alpha_vantage,
            self._finnhub
        ]
    
    def get_next_twelvedata_key(self) -> str:
        """Rotate through TwelveData keys for rate limiting"""
        if not self.twelvedata_keys:
            raise ValueError("No TwelveData keys configured")
        
        key = self.twelvedata_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.twelvedata_keys)
        return key
    
    async def get_forex_data(self, symbol: str, interval: str = "1min", limit: int = 100) -> Optional[Dict]:
        """Get market data with smart fallbacks"""
        
        # Try TwelveData first (your primary)
        data = await self._twelvedata(symbol, interval, limit)
        if data:
            logger.info(f"✅ Data from TwelveData for {symbol}")
            return data
        
        # Fallback to free APIs if TwelveData fails
        for fallback in self.fallback_apis:
            data = await fallback(symbol, interval, limit)
            if data:
                logger.info(f"✅ Data from fallback for {symbol}")
                return data
        
        logger.error(f"❌ All APIs failed for {symbol}")
        return None
    
    async def _twelvedata(self, symbol: str, interval: str, limit: int) -> Optional[Dict]:
        """TwelveData API call with key rotation"""
        try:
            key = self.get_next_twelvedata_key()
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
                return {
                    "provider": "twelvedata",
                    "symbol": symbol,
                    "data": data["values"],
                    "meta": data.get("meta", {})
                }
            
            logger.warning(f"TwelveData no data: {data.get('message', 'Unknown error')}")
            return None
            
        except Exception as e:
            logger.error(f"TwelveData error: {e}")
            return None
    
    async def _yahoo_finance(self, symbol: str, interval: str, limit: int) -> Optional[Dict]:
        """Yahoo Finance fallback (FREE, no key)"""
        try:
            # Convert symbol format: EUR/USD -> EURUSD=X
            yahoo_symbol = symbol.replace("/", "") + "=X"
            
            import yfinance as yf
            data = yf.download(yahoo_symbol, period="1d", interval="1m")
            
            if not data.empty:
                return {
                    "provider": "yahoo",
                    "symbol": symbol,
                    "data": data.tail(limit).to_dict('records')
                }
            return None
            
        except Exception as e:
            logger.warning(f"Yahoo Finance error: {e}")
            return None
    
    async def _alpha_vantage(self, symbol: str, interval: str, limit: int) -> Optional[Dict]:
        """Alpha Vantage fallback"""
        if not Config.ALPHAVANTAGE_API_KEY:
            return None
            
        try:
            # Alpha Vantage implementation here
            pass
        except Exception as e:
            logger.warning(f"Alpha Vantage error: {e}")
            return None
    
    async def _finnhub(self, symbol: str, interval: str, limit: int) -> Optional[Dict]:
        """Finnhub fallback"""
        if not Config.FINNHUB_API_KEY:
            return None
            
        try:
            # Finnhub implementation here
            pass
        except Exception as e:
            logger.warning(f"Finnhub error: {e}")
            return None

# Global instance
market_data = MarketDataAPI()
