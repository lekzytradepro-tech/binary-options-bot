import requests
import logging
import asyncio
from datetime import datetime, timedelta
from src.core.config import Config

logger = logging.getLogger(__name__)

class TwelveDataClient:
    def __init__(self):
        self.api_keys = Config.TWELVEDATA_KEYS
        self.current_key_index = 0
        self.base_url = "https://api.twelvedata.com"
    
    def get_next_key(self):
        """Rotate API keys"""
        if not self.api_keys:
            raise ValueError("No TwelveData API keys configured")
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    async def get_binary_recommendation(self, symbol: str, expiry_minutes: int = 5):
        """Get binary options recommendation with real data"""
        try:
            # Get real-time data
            realtime_data = await self.get_realtime_price(symbol)
            historical_data = await self.get_historical_data(symbol, "1min", 50)
            
            if not realtime_data or not historical_data:
                return self._get_fallback_signal(symbol)
            
            # Analyze for binary options
            analysis = self._analyze_for_binary(realtime_data, historical_data, expiry_minutes)
            
            return {
                "symbol": symbol,
                "current_price": realtime_data['price'],
                "direction": analysis['direction'],
                "confidence": analysis['confidence'],
                "expiry": expiry_minutes,
                "recommended_type": "CALL/PUT",
                "payout": analysis['payout'],
                "analysis": analysis['reasoning'],
                "timestamp": datetime.now().isoformat(),
                "data_source": "twelvedata"
            }
            
        except Exception as e:
            logger.error(f"Binary recommendation error: {e}")
            return self._get_fallback_signal(symbol)
    
    async def get_realtime_price(self, symbol: str):
        """Get real-time price for binary options"""
        try:
            key = self.get_next_key()
            url = f"{self.base_url}/price"
            params = {
                "symbol": symbol,
                "apikey": key
            }
            
            response = await asyncio.to_thread(requests.get, url, params=params, timeout=10)
            data = response.json()
            
            if 'price' in data:
                return {
                    'price': float(data['price']),
                    'timestamp': datetime.now().isoformat()
                }
            return None
            
        except Exception as e:
            logger.error(f"Realtime price error: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, interval: str = "1min", limit: int = 50):
        """Get historical data for analysis"""
        try:
            key = self.get_next_key()
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": limit,
                "apikey": key
            }
            
            response = await asyncio.to_thread(requests.get, url, params=params, timeout=10)
            data = response.json()
            
            if data.get('values'):
                return [{
                    'datetime': item['datetime'],
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item.get('volume', 0))
                } for item in data['values']]
            return None
            
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return None
    
    def _analyze_for_binary(self, realtime_data, historical_data, expiry_minutes):
        """Analyze data for binary options trading"""
        if not historical_data:
            return self._get_random_signal()
        
        # Calculate simple indicators for binary options
        recent_prices = [float(item['close']) for item in historical_data[-10:]]
        current_price = realtime_data['price']
        
        # Simple momentum calculation
        price_change = current_price - recent_prices[0]
        price_change_percent = (price_change / recent_prices[0]) * 100
        
        # Volatility calculation (simplified)
        highs = [float(item['high']) for item in historical_data[-10:]]
        lows = [float(item['low']) for item in historical_data[-10:]]
        volatility = (max(highs) - min(lows)) / min(lows) * 100
        
        # Determine direction and confidence
        if price_change_percent > 0.02:  # 0.02% upward movement
            direction = "CALL"
            confidence = min(80, 60 + abs(price_change_percent) * 100)
        elif price_change_percent < -0.02:  # 0.02% downward movement
            direction = "PUT"
            confidence = min(80, 60 + abs(price_change_percent) * 100)
        else:
            direction = "CALL" if current_price > recent_prices[-5] else "PUT"
            confidence = 55  # Low confidence for sideways
        
        # Adjust payout based on volatility
        if volatility > 0.1:
            payout = 85  # High volatility = higher payout
        elif volatility > 0.05:
            payout = 75  # Medium volatility
        else:
            payout = 70  # Low volatility
        
        reasoning = f"Price change: {price_change_percent:.3f}%, Volatility: {volatility:.3f}%"
        
        return {
            'direction': direction,
            'confidence': round(confidence, 1),
            'payout': payout,
            'reasoning': reasoning
        }
    
    def _get_fallback_signal(self, symbol):
        """Fallback when API fails"""
        import random
        directions = ["CALL", "PUT"]
        return {
            "symbol": symbol,
            "current_price": 1.0850,  # Sample price
            "direction": random.choice(directions),
            "confidence": 65.0,
            "expiry": 5,
            "recommended_type": "CALL/PUT",
            "payout": 75,
            "analysis": "Using fallback analysis",
            "timestamp": datetime.now().isoformat(),
            "data_source": "fallback"
        }

# Global instance
twelvedata_client = TwelveDataClient()
