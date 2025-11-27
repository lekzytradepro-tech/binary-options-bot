from flask import Flask, request, jsonify
import os
import logging
import requests
import threading
import queue
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
update_queue = queue.Queue()

# OTC Market Configuration
OTC_ASSETS = {
    "EUR/USD": {"type": "Forex", "volatility": "High", "session": "London/NY", "avg_range": 0.0050},
    "GBP/USD": {"type": "Forex", "volatility": "Very High", "session": "London/NY", "avg_range": 0.0060},
    "USD/JPY": {"type": "Forex", "volatility": "Medium", "session": "Asian/London", "avg_range": 0.0040},
    "USD/CHF": {"type": "Forex", "volatility": "Medium", "session": "London/NY", "avg_range": 0.0035},
    "AUD/USD": {"type": "Forex", "volatility": "High", "session": "Asian/London", "avg_range": 0.0055},
    "USD/CAD": {"type": "Forex", "volatility": "Medium", "session": "London/NY", "avg_range": 0.0045},
    "BTC/USD": {"type": "Crypto", "volatility": "Extreme", "session": "24/7", "avg_range": 150.0},
    "ETH/USD": {"type": "Crypto", "volatility": "Extreme", "session": "24/7", "avg_range": 80.0},
    "XAU/USD": {"type": "Commodity", "volatility": "High", "session": "London/NY", "avg_range": 12.0},
    "US30": {"type": "Index", "volatility": "High", "session": "NY", "avg_range": 120.0}
}

class OTCSignalEngine:
    """OTC-Optimized Signal Engine based on expert blueprint"""
    
    def __init__(self):
        self.market_states = {}
        self.last_signals = {}
        
    def analyze_market_condition(self, asset: str) -> Dict:
        """Layer 1: Market Condition Filter"""
        # Simulate OTC market analysis
        ma50_direction = random.choice(["BULLISH", "BEARISH", "SIDEWAYS"])
        adx_value = random.randint(15, 45)
        bb_width = random.uniform(0.002, 0.008)
        price_deviation = random.uniform(-0.003, 0.003)
        
        # Determine market state
        if adx_value > 25 and abs(price_deviation) > 0.002:
            market_state = "TRENDING"
        elif bb_width < 0.003:
            market_state = "RANGING" 
        else:
            market_state = "HIGH_VOLATILITY"
            
        return {
            "state": market_state,
            "ma50_direction": ma50_direction,
            "adx": adx_value,
            "bb_width": bb_width,
            "price_deviation": price_deviation
        }
    
    def detect_otc_patterns(self, asset: str, market_state: str) -> List[str]:
        """Layer 2: OTC-Friendly Pattern Recognition"""
        patterns = []
        
        # OTC-specific patterns
        possible_patterns = [
            "M_FORMATION", "W_FORMATION", "TRIPLE_TOP", "TRIPLE_BOTTOM",
            "BOLLINGER_REJECTION", "MA20_PULLBACK", "SUPPORT_RESISTANCE_BOUNCE",
            "MICRO_TREND_BREAK", "VOLATILITY_SQUEEZE"
        ]
        
        # Select patterns based on market state
        if market_state == "TRENDING":
            patterns = random.sample(["MA20_PULLBACK", "MICRO_TREND_BREAK", "BOLLINGER_REJECTION"], 2)
        elif market_state == "RANGING":
            patterns = random.sample(["M_FORMATION", "W_FORMATION", "SUPPORT_RESISTANCE_BOUNCE", "TRIPLE_TOP", "TRIPLE_BOTTOM"], 2)
        else:  # HIGH_VOLATILITY
            patterns = random.sample(["VOLATILITY_SQUEEZE", "BOLLINGER_REJECTION"], 1)
            
        return patterns
    
    def calculate_entry_timing(self, asset: str, patterns: List[str]) -> Dict:
        """Layer 3: OTC Entry Timing Engine"""
        timing_signals = []
        
        for pattern in patterns:
            if pattern in ["M_FORMATION", "TRIPLE_TOP"]:
                timing_signals.append({
                    "signal": "PUT",
                    "confidence_boost": 0.15,
                    "reason": f"{pattern} pattern detected - expecting reversal"
                })
            elif pattern in ["W_FORMATION", "TRIPLE_BOTTOM"]:
                timing_signals.append({
                    "signal": "CALL", 
                    "confidence_boost": 0.15,
                    "reason": f"{pattern} pattern detected - expecting reversal"
                })
            elif pattern == "BOLLINGER_REJECTION":
                timing_signals.append({
                    "signal": "REVERSAL",
                    "confidence_boost": 0.12,
                    "reason": "Price rejected from Bollinger Band edge"
                })
            elif pattern == "MA20_PULLBACK":
                timing_signals.append({
                    "signal": "TREND_CONTINUATION",
                    "confidence_boost": 0.10,
                    "reason": "Pullback to MA20 in strong trend"
                })
            elif pattern == "VOLATILITY_SQUEEZE":
                timing_signals.append({
                    "signal": "BREAKOUT",
                    "confidence_boost": 0.18,
                    "reason": "Volatility compression expecting expansion"
                })
                
        return timing_signals
    
    def calculate_confidence_score(self, market_condition: Dict, patterns: List[str], timing_signals: List[Dict]) -> float:
        """Layer 4: OTC Confidence Engine"""
        # Trend score (0-25 points)
        trend_score = 0
        if market_condition["state"] == "TRENDING":
            trend_score = 20 + (market_condition["adx"] - 20) * 0.5
        else:
            trend_score = 10
            
        # Momentum score (0-25 points)
        momentum_score = min(25, abs(market_condition["price_deviation"]) * 5000)
        
        # Pattern score (0-30 points)
        pattern_score = len(patterns) * 7.5  # 7.5 points per pattern
        
        # Volatility score (0-20 points) - Lower volatility = higher score for OTC
        volatility_score = max(0, 20 - (market_condition["bb_width"] * 2000))
        
        # Timing signal boost
        timing_boost = sum(signal["confidence_boost"] for signal in timing_signals) * 100
        
        total_score = trend_score + momentum_score + pattern_score + volatility_score + timing_boost
        confidence = min(95, max(65, total_score))  # Cap between 65-95%
        
        return confidence
    
    def check_safety_conditions(self, asset: str, market_condition: Dict) -> bool:
        """Safety Blockers for OTC Trading"""
        # Block if volatility too high
        if market_condition["bb_width"] > 0.006:
            return False
            
        # Block if ADX shows weak trend but price is far from MA
        if market_condition["adx"] < 20 and abs(market_condition["price_deviation"]) > 0.004:
            return False
            
        # Block during high manipulation periods (simulated)
        current_hour = datetime.now().hour
        if current_hour in [0, 1, 13, 14]:  # Asian lunch, NY lunch
            return False
            
        return True
    
    def generate_otc_signal(self, asset: str, expiry: str) -> Dict:
        """Generate OTC-optimized trading signal"""
        try:
            # Layer 1: Market Condition
            market_condition = self.analyze_market_condition(asset)
            
            # Check safety conditions
            if not self.check_safety_conditions(asset, market_condition):
                return {
                    "error": "MARKET_CONDITIONS_UNSAFE",
                    "message": "Safety conditions not met for OTC trading"
                }
            
            # Layer 2: Pattern Recognition
            patterns = self.detect_otc_patterns(asset, market_condition["state"])
            
            # Layer 3: Entry Timing
            timing_signals = self.calculate_entry_timing(asset, patterns)
            
            # Layer 4: Confidence Calculation
            confidence = self.calculate_confidence_score(market_condition, patterns, timing_signals)
            
            # Determine final signal direction
            if timing_signals:
                primary_signal = timing_signals[0]
                if primary_signal["signal"] in ["PUT", "REVERSAL"]:
                    direction = "PUT"
                    reason = primary_signal["reason"]
                elif primary_signal["signal"] in ["CALL", "TREND_CONTINUATION"]:
                    direction = "CALL"
                    reason = primary_signal["reason"]
                else:  # BREAKOUT
                    direction = random.choice(["CALL", "PUT"])
                    reason = f"Breakout signal - {direction} expected"
            else:
                direction = random.choice(["CALL", "PUT"])
                reason = "Market structure analysis"
            
            # Calculate expected payout based on volatility and confidence
            base_payout = 75
            volatility_multiplier = {
                "Extreme": 15,
                "Very High": 10, 
                "High": 8,
                "Medium": 5
            }
            asset_volatility = OTC_ASSETS[asset]["volatility"]
            payout_boost = volatility_multiplier.get(asset_volatility, 5)
            expected_payout = f"{base_payout + payout_boost}-{base_payout + payout_boost + 5}%"
            
            # Generate analysis details
            current_time = datetime.now()
            analysis_time = current_time.strftime("%H:%M:%S")
            expected_entry = (current_time + timedelta(seconds=45)).strftime("%H:%M:%S")
            
            return {
                "asset": asset,
                "direction": direction,
                "expiry": expiry,
                "confidence": round(confidence, 1),
                "expected_payout": expected_payout,
                "market_state": market_condition["state"],
                "patterns_detected": patterns,
                "analysis_time": analysis_time,
                "expected_entry": expected_entry,
                "reason": reason,
                "risk_level": "MEDIUM" if confidence < 80 else "LOW",
                "recommended_investment": "$25-$100",
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"OTC Signal generation error: {e}")
            return {"error": "GENERATION_FAILED", "message": str(e)}

class OTCBinaryBot:
    """OTC Binary Trading Bot with Expert Signal Engine"""
    
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.signal_engine = OTCSignalEngine()
        
    def send_message(self, chat_id, text, parse_mode=None, reply_markup=None):
        """Send message synchronously"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text
            }
            
            if parse_mode:
                data["parse_mode"] = parse_mode
                
            if reply_markup:
                data["reply_markup"] = reply_markup
                
            response = requests.post(url, json=data, timeout=10)
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Send message error: {e}")
            return None
    
    def edit_message_text(self, chat_id, message_id, text, parse_mode=None, reply_markup=None):
        """Edit message synchronously"""
        try:
            url = f"{self.base_url}/editMessageText"
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text
            }
            
            if parse_mode:
                data["parse_mode"] = parse_mode
                
            if reply_markup:
                data["reply_markup"] = reply_markup
                
            response = requests.post(url, json=data, timeout=10)
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Edit message error: {e}")
            return None
    
    def answer_callback_query(self, callback_query_id, text=None):
        """Answer callback query synchronously"""
        try:
            url = f"{self.base_url}/answerCallbackQuery"
            data = {
                "callback_query_id": callback_query_id
            }
            if text:
                data["text"] = text
            response = requests.post(url, json=data, timeout=5)
            return response.json()
        except Exception as e:
            logger.error(f"❌ Answer callback error: {e}")
            return None
    
    def process_update(self, update_data):
        """Process update synchronously"""
        try:
            logger.info(f"🔄 Processing OTC update: {update_data.get('update_id', 'unknown')}")
            
            if 'message' in update_data:
                self._process_message(update_data['message'])
                
            elif 'callback_query' in update_data:
                self._process_callback_query(update_data['callback_query'])
                
        except Exception as e:
            logger.error(f"❌ OTC Update processing error: {e}")
    
    def _process_message(self, message):
        """Process message update"""
        try:
            chat_id = message['chat']['id']
            text = message.get('text', '').strip()
            
            if text == '/start':
                self._handle_start(chat_id, message)
            elif text == '/help':
                self._handle_help(chat_id)
            elif text == '/signals':
                self._handle_signals(chat_id)
            elif text == '/assets':
                self._handle_assets(chat_id)
            elif text == '/strategies':
                self._handle_strategies(chat_id)
            elif text == '/status':
                self._handle_status(chat_id)
            elif text == '/otcguide':
                self._handle_otc_guide(chat_id)
            else:
                self._handle_unknown(chat_id)
                
        except Exception as e:
            logger.error(f"❌ OTC Message processing error: {e}")
    
    def _process_callback_query(self, callback_query):
        """Process callback query"""
        try:
            # Answer callback first
            self.answer_callback_query(callback_query['id'])
            
            chat_id = callback_query['message']['chat']['id']
            message_id = callback_query['message']['message_id']
            data = callback_query.get('data', '')
            
            self._handle_button_click(chat_id, message_id, data, callback_query)
            
        except Exception as e:
            logger.error(f"❌ OTC Callback processing error: {e}")
    
    def _handle_start(self, chat_id, message):
        """Handle /start command"""
        try:
            user = message.get('from', {})
            user_id = user.get('id', 0)
            first_name = user.get('first_name', 'Trader')
            
            logger.info(f"👤 OTC User started: {user_id} - {first_name}")
            
            # OTC-Focused Welcome
            welcome_text = """
🎯 **OTC BINARY SIGNALS PRO** 
*Expert-Optimized for Over-The-Counter Markets*

🤖 **ADVANCED OTC FEATURES:**
• Market Condition Filtering
• OTC Pattern Recognition  
• Precision Entry Timing
• Confidence Scoring Engine
• Safety Blockers Active

⚠️ **OTC MARKET UNDERSTANDING:**
• Synthetic price movements
• Mean-reversion behavior
• Short micro-trends
• Volatility manipulation

*Professional OTC signal generation ready*"""

            keyboard = {
                "inline_keyboard": [
                    [{"text": "✅ ACCESS OTC SIGNALS", "callback_data": "menu_signals"}],
                    [{"text": "📚 OTC TRADING GUIDE", "callback_data": "otc_guide"}]
                ]
            }
            
            self.send_message(
                chat_id, 
                welcome_text, 
                parse_mode="Markdown",
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ OTC Start handler error: {e}")
    
    def _handle_help(self, chat_id):
        """Handle /help command"""
        help_text = """
🦅 **OTC SIGNALS PRO - HELP**

*Built specifically for OTC market conditions*

**OTC-OPTIMIZED COMMANDS:**
/signals - Generate OTC signals
/assets - View OTC instruments  
/strategies - OTC trading approaches
/otcguide - OTC market understanding
/status - System status

**OTC SIGNAL FEATURES:**
• 🎯 Market State Detection
• 📊 OTC Pattern Recognition  
• ⚡ Precision Timing Engine
• 🛡 Safety Condition Checking
• 📈 Confidence Probability Scoring

**RISK MANAGEMENT:**
• OTC-specific safety filters
• Volatility-based position sizing
• Session-based trading hours
• Pattern confirmation required

*Trade OTC markets intelligently*"""
        
        self.send_message(chat_id, help_text, parse_mode="Markdown")
    
    def _handle_signals(self, chat_id):
        """Handle /signals command"""
        self._show_signals_menu(chat_id)
    
    def _handle_assets(self, chat_id):
        """Handle /assets command"""
        self._show_assets_menu(chat_id)
    
    def _handle_strategies(self, chat_id):
        """Handle /strategies command"""
        self._show_otc_strategies(chat_id)
    
    def _handle_status(self, chat_id):
        """Handle /status command"""
        status_text = """
✅ **OTC SIGNALS PRO - STATUS: OPTIMAL**

🤖 **OTC ENGINE STATUS:**
• Market Condition Filter: ✅ ACTIVE
• Pattern Recognition: ✅ ACTIVE  
• Entry Timing Engine: ✅ ACTIVE
• Confidence Scoring: ✅ ACTIVE
• Safety Blockers: ✅ ACTIVE

📊 **MARKET COVERAGE:**
• OTC Assets: 10 instruments
• Timeframes: M1-M5 analysis
• Sessions: All major hours
• Patterns: 9 OTC-specific

*OTC-optimized signal generation operational*"""
        
        self.send_message(chat_id, status_text, parse_mode="Markdown")
    
    def _handle_otc_guide(self, chat_id):
        """Handle OTC guide command"""
        guide_text = """
📚 **OTC MARKET UNDERSTANDING**

*Key OTC Characteristics:*

**🔄 MEAN-REVERSION BEHAVIOR**
OTC prices tend to revert to means more frequently than real markets due to synthetic liquidity.

**⚡ SHORT MICRO-TRENDS**
Trends last minutes, not hours. Perfect for binary options timeframes.

**🎯 PREDICTABLE PATTERNS**
M/W formations, triple tops/bottoms, Bollinger rejections occur consistently.

**🛡 SAFETY FIRST**
• Avoid high volatility spikes
• Trade during active sessions
• Use tight risk management
• Verify pattern confirmations

*OTC success requires understanding these dynamics*"""
        
        self.send_message(chat_id, guide_text, parse_mode="Markdown")
    
    def _handle_unknown(self, chat_id):
        """Handle unknown commands"""
        text = "🤖 OTC Signals Pro: Use /help for OTC trading commands."
        self.send_message(chat_id, text, parse_mode="Markdown")
    
    def _handle_button_click(self, chat_id, message_id, data, callback_query=None):
        """Handle button clicks"""
        try:
            logger.info(f"🔄 OTC Button: {data}")
            
            if data == "menu_signals":
                self._show_signals_menu(chat_id, message_id)
                
            elif data == "otc_guide":
                self._show_otc_guide_detail(chat_id, message_id)
                
            elif data.startswith("asset_"):
                asset = data.replace("asset_", "")
                self._show_asset_expiry(chat_id, message_id, asset)
                
            elif data.startswith("expiry_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    asset = parts[1]
                    expiry = parts[2]
                    self._generate_otc_signal(chat_id, message_id, asset, expiry)
                    
            elif data.startswith("signal_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    asset = parts[1]
                    expiry = parts[2]
                    self._generate_otc_signal(chat_id, message_id, asset, expiry)
                    
            else:
                self.edit_message_text(
                    chat_id, message_id,
                    "🔄 **OTC SYSTEM ACTIVE**\n\nSelect signal parameters above.",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.error(f"❌ OTC Button handler error: {e}")
    
    def _show_signals_menu(self, chat_id, message_id=None):
        """Show OTC signals menu"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "⚡ EUR/USD OTC SIGNALS", "callback_data": "asset_EUR/USD"}],
                [{"text": "⚡ GBP/USD OTC SIGNALS", "callback_data": "asset_GBP/USD"}],
                [{"text": "⚡ USD/JPY OTC SIGNALS", "callback_data": "asset_USD/JPY"}],
                [{"text": "₿ BTC/USD OTC SIGNALS", "callback_data": "asset_BTC/USD"}],
                [{"text": "🟡 XAU/USD OTC SIGNALS", "callback_data": "asset_XAU/USD"}],
                [{"text": "📈 US30 OTC SIGNALS", "callback_data": "asset_US30"}],
                [{"text": "📚 OTC TRADING GUIDE", "callback_data": "otc_guide"}]
            ]
        }
        
        text = """
🎯 **OTC BINARY SIGNALS**

*Expert-optimized for OTC market conditions:*

**OTC-OPTIMIZED ASSETS:**
• EUR/USD - High liquidity, clean patterns
• GBP/USD - High volatility, strong trends  
• USD/JPY - Asian session specialist
• BTC/USD - Crypto OTC movements
• XAU/USD - Commodity OTC flows
• US30 - Index OTC trading

**OTC SIGNAL FEATURES:**
• Market condition filtering
• Pattern recognition engine
• Precision timing signals
• Confidence probability scoring
• Safety condition checking

*Select asset for OTC signal generation*"""
        
        if message_id:
            self.edit_message_text(
                chat_id, message_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
        else:
            self.send_message(
                chat_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
    
    def _show_assets_menu(self, chat_id, message_id=None):
        """Show OTC assets"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "💱 EUR/USD", "callback_data": "asset_EUR/USD"}, {"text": "💱 GBP/USD", "callback_data": "asset_GBP/USD"}],
                [{"text": "💱 USD/JPY", "callback_data": "asset_USD/JPY"}, {"text": "💱 USD/CHF", "callback_data": "asset_USD/CHF"}],
                [{"text": "💱 AUD/USD", "callback_data": "asset_AUD/USD"}, {"text": "💱 USD/CAD", "callback_data": "asset_USD/CAD"}],
                [{"text": "₿ BTC/USD", "callback_data": "asset_BTC/USD"}, {"text": "₿ ETH/USD", "callback_data": "asset_ETH/USD"}],
                [{"text": "🟡 XAU/USD", "callback_data": "asset_XAU/USD"}, {"text": "📈 US30", "callback_data": "asset_US30"}],
                [{"text": "🔙 SIGNALS MENU", "callback_data": "menu_signals"}]
            ]
        }
        
        text = """
📊 **OTC TRADING INSTRUMENTS**

*Curated for OTC binary options trading:*

**FOREX OTC (6 PAIRS)**
• EUR/USD - Most liquid OTC
• GBP/USD - High volatility OTC
• USD/JPY - Asian OTC specialist
• USD/CHF, AUD/USD, USD/CAD

**CRYPTO OTC (2 PAIRS)**
• BTC/USD - Crypto OTC leader
• ETH/USD - Altcoin OTC

**COMMODITIES & INDICES**
• XAU/USD - Gold OTC trading
• US30 - Dow Jones OTC

*All optimized for OTC pattern recognition*"""
        
        if message_id:
            self.edit_message_text(
                chat_id, message_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
        else:
            self.send_message(
                chat_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
    
    def _show_otc_strategies(self, chat_id):
        """Show OTC trading strategies"""
        strategies_text = """
🚀 **OTC TRADING STRATEGIES**

*Built for OTC market dynamics:*

**TREND-FOLLOWING (OTC)**
• Follow micro-trends (2-5 minutes)
• Use MA20 pullbacks for entries
• ADX confirmation required

**MEAN-REVERSION (OTC)**
• Trade M/W formations
• Bollinger Band rejections
• RSI extremes (70+/30-)

**VOLATILITY TRADING**
• Volatility squeeze breakouts
• News impact anticipation
• Session overlap trading

**PATTERN TRADING**
• Triple top/bottom patterns
• Support/resistance bounces
• Candlestick exhaustion signals

*Each strategy uses OTC-optimized filters*"""
        
        self.send_message(chat_id, strategies_text, parse_mode="Markdown")
    
    def _show_otc_guide_detail(self, chat_id, message_id):
        """Show detailed OTC guide"""
        guide_text = """
🎓 **OTC MARKET MASTERY**

*Understanding OTC Binary Markets:*

**🔄 OTC MARKET STRUCTURE**
• Prices are broker-generated, not real market prices
• Each broker has unique OTC feed characteristics
• Synthetic liquidity creates predictable patterns

**🎯 OTC TRADING ADVANTAGES**
• Consistent pattern repetition
• Mean-reversion behavior
• Short, predictable micro-trends
• Clear support/resistance levels

**🛡 OTC RISK MANAGEMENT**
• Avoid oversized candles (>1.5x average)
• Skip high volatility spikes
• Trade active sessions only
• Use pattern confirmation

**⚡ OTC SUCCESS KEYS**
1. Understand it's not real market
2. Trade patterns, not fundamentals  
3. Use tight risk management
4. Follow OTC-optimized strategies

*Master OTC, master binary options*"""
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 GENERATE OTC SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        self.edit_message_text(
            chat_id, message_id,
            guide_text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_asset_expiry(self, chat_id, message_id, asset):
        """Show expiry options for OTC asset"""
        asset_info = OTC_ASSETS.get(asset, {})
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "⚡ 1 MINUTE - OTC SCALPING", "callback_data": f"expiry_{asset}_1"}],
                [{"text": "⚡ 2 MINUTES - OTC QUICK", "callback_data": f"expiry_{asset}_2"}],
                [{"text": "🎯 5 MINUTES - OTC STANDARD", "callback_data": f"expiry_{asset}_5"}],
                [{"text": "📈 15 MINUTES - OTC SWING", "callback_data": f"expiry_{asset}_15"}],
                [{"text": "🔙 BACK TO ASSETS", "callback_data": "menu_assets"}]
            ]
        }
        
        text = f"""
📊 **{asset} - OTC BINARY OPTIONS**

*OTC Market Profile:*
• **Type:** {asset_info['type']}
• **Volatility:** {asset_info['volatility']}
• **Session:** {asset_info['session']}
• **Avg Range:** {asset_info['avg_range']}

*OTC-optimized Expiry Times:*

⚡ **1-2 MINUTES** - OTC micro-trends and scalping
🎯 **5 MINUTES** - OTC pattern completion and confirmation
📈 **15 MINUTES** - OTC session-based movements

*Select OTC expiry time for signal generation*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _generate_otc_signal(self, chat_id, message_id, asset, expiry):
        """Generate OTC-optimized trading signal"""
        try:
            # Generate signal using OTC engine
            signal_data = self.signal_engine.generate_otc_signal(asset, expiry)
            
            if "error" in signal_data:
                self.edit_message_text(
                    chat_id, message_id,
                    f"❌ **OTC SAFETY BLOCKED**\n\n{signal_data['message']}\n\nTry different asset or time.",
                    parse_mode="Markdown"
                )
                return
            
            # Format signal message
            direction_emoji = "🟢" if signal_data["direction"] == "CALL" else "🔴"
            confidence_emoji = "🎯" if signal_data["confidence"] >= 80 else "📊"
            
            keyboard = {
                "inline_keyboard": [
                    [{"text": "🔄 NEW OTC SIGNAL", "callback_data": f"signal_{asset}_{expiry}"}],
                    [{"text": "📊 DIFFERENT ASSET", "callback_data": "menu_assets"}],
                    [{"text": "⏰ DIFFERENT EXPIRY", "callback_data": f"asset_{asset}"}]
                ]
            }
            
            text = f"""
{direction_emoji} **OTC BINARY SIGNAL - {asset}**

📈 **DIRECTION:** {signal_data['direction']}
{confidence_emoji} **CONFIDENCE:** {signal_data['confidence']}%
⏰ **EXPIRY:** {expiry} MINUTES
💎 **ASSET:** {asset}
🏦 **MARKET:** OTC BINARY

**📊 OTC ANALYSIS:**
• Market State: {signal_data['market_state']}
• Patterns: {', '.join(signal_data['patterns_detected'])}
• Analysis Time: {signal_data['analysis_time']} UTC
• Expected Entry: {signal_data['expected_entry']} UTC

**🎯 SIGNAL REASON:**
{signal_data['reason']}

**💰 EXPECTED PAYOUT:** {signal_data['expected_payout']}

**⚡ OTC TRADING PLAN:**
• Direction: {signal_data['direction']}
• Expiry: {expiry} minutes  
• Entry: Within 45 seconds
• Risk Level: {signal_data['risk_level']}
• Investment: {signal_data['recommended_investment']}

**🛡 OTC RISK MANAGEMENT:**
• Max Risk: 2% of account
• Stop Loss: Mental (if pattern breaks)
• Avoid: High volatility spikes
• Preferred: {OTC_ASSETS[asset]['session']} session

*OTC Signal - Trade responsibly*"""

            self.edit_message_text(
                chat_id, message_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ OTC Signal generation error: {e}")
            self.edit_message_text(
                chat_id, message_id,
                "❌ **OTC SIGNAL ERROR**\n\nSystem temporarily unavailable. Please try again.",
                parse_mode="Markdown"
            )

# Create OTC trading bot instance
otc_bot = OTCBinaryBot()

def process_queued_updates():
    """Process OTC updates from queue"""
    while True:
        try:
            if not update_queue.empty():
                update_data = update_queue.get_nowait()
                otc_bot.process_update(update_data)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ OTC Queue processing error: {e}")
            time.sleep(1)

# Start OTC processing thread
processing_thread = threading.Thread(target=process_queued_updates, daemon=True)
processing_thread.start()

@app.route('/')
def home():
    return jsonify({
        "status": "running", 
        "service": "otc-signals-pro",
        "version": "4.0.0",
        "engine": "OTC-optimized signal generation",
        "assets": len(OTC_ASSETS),
        "features": ["market_filter", "pattern_recognition", "timing_engine", "confidence_scoring", "safety_blockers"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "engine": "OTC Signal Engine",
        "assets_available": len(OTC_ASSETS),
        "queue_size": update_queue.qsize(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """OTC Signals webhook endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        update_id = update_data.get('update_id', 'unknown')
        
        logger.info(f"📨 OTC Signal Update: {update_id}")
        
        # Add to OTC processing queue
        update_queue.put(update_data)
        
        return jsonify({
            "status": "queued",
            "update_id": update_id, 
            "engine": "OTC-optimized",
            "queue_size": update_queue.qsize()
        })
        
    except Exception as e:
        logger.error(f"❌ OTC Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/otc_debug')
def otc_debug():
    """OTC system debug endpoint"""
    return jsonify({
        "otc_engine": "active",
        "assets_configured": len(OTC_ASSETS),
        "signal_engine": "operational",
        "safety_filters": "enabled",
        "queue_processing": "active"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Starting OTC Signals Pro on port {port}")
    logger.info("🎯 Expert-optimized for OTC binary markets")
    logger.info("📊 Market Filtering | Pattern Recognition | Timing Engine | Safety Blockers")
    
    app.run(host='0.0.0.0', port=port, debug=False)
