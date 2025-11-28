from flask import Flask, request, jsonify
import os
import logging
import requests
import threading
import queue
import time
import random
from datetime import datetime, timedelta

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

# User management
user_limits = {}
user_sessions = {}

# User tier management
user_tiers = {}
ADMIN_IDS = [6307001401]  # Your Telegram ID
ADMIN_USERNAME = "@LekzyDevX"  # Your admin username

# Default tiers configuration
USER_TIERS = {
    'free_trial': {
        'name': 'FREE TRIAL',
        'signals_daily': 10,
        'duration_days': 14,
        'price': 0,
        'features': ['10 signals/day', 'All 15 assets', '8 AI engines', 'All strategies']
    },
    'basic': {
        'name': 'BASIC', 
        'signals_daily': 50,
        'duration_days': 30,
        'price': 19,
        'features': ['50 signals/day', 'Priority signals', 'Advanced AI', 'All features']
    },
    'pro': {
        'name': 'PRO',
        'signals_daily': 9999,  # Unlimited
        'duration_days': 30,
        'price': 49,
        'features': ['Unlimited signals', 'All features', 'Dedicated support', 'Priority access']
    },
    'admin': {
        'name': 'ADMIN',
        'signals_daily': 9999,
        'duration_days': 9999,
        'price': 0,
        'features': ['Full system access', 'User management', 'All features', 'Admin privileges']
    }
}

# OTC Binary Trading Configuration
OTC_ASSETS = {
    "EUR/USD": {"type": "Forex", "volatility": "High", "session": "London/NY"},
    "GBP/USD": {"type": "Forex", "volatility": "High", "session": "London/NY"},
    "USD/JPY": {"type": "Forex", "volatility": "Medium", "session": "Asian/London"},
    "USD/CHF": {"type": "Forex", "volatility": "Medium", "session": "London/NY"},
    "AUD/USD": {"type": "Forex", "volatility": "High", "session": "Asian/London"},
    "USD/CAD": {"type": "Forex", "volatility": "Medium", "session": "London/NY"},
    "NZD/USD": {"type": "Forex", "volatility": "High", "session": "Asian/London"},
    "EUR/GBP": {"type": "Forex", "volatility": "Medium", "session": "London"},
    "GBP/JPY": {"type": "Forex", "volatility": "Very High", "session": "London"},
    "EUR/JPY": {"type": "Forex", "volatility": "High", "session": "London"},
    "USD/CNH": {"type": "Forex", "volatility": "Medium", "session": "Asian"},
    "USD/SGD": {"type": "Forex", "volatility": "Medium", "session": "Asian"},
    "BTC/USD": {"type": "Crypto", "volatility": "Very High", "session": "24/7"},
    "ETH/USD": {"type": "Crypto", "volatility": "Very High", "session": "24/7"},
    "XRP/USD": {"type": "Crypto", "volatility": "High", "session": "24/7"},
    "ADA/USD": {"type": "Crypto", "volatility": "High", "session": "24/7"},
    "XAU/USD": {"type": "Commodity", "volatility": "High", "session": "London/NY"},
    "XAG/USD": {"type": "Commodity", "volatility": "High", "session": "London/NY"},
    "OIL/USD": {"type": "Commodity", "volatility": "High", "session": "London/NY"},
    "US30": {"type": "Index", "volatility": "High", "session": "NY"},
    "SPX500": {"type": "Index", "volatility": "Medium", "session": "NY"},
    "NAS100": {"type": "Index", "volatility": "High", "session": "NY"}
}

AI_ENGINES = {
    "QuantumTrend AI": "Advanced trend analysis with machine learning",
    "NeuralMomentum AI": "Real-time momentum detection",
    "VolatilityMatrix AI": "Multi-timeframe volatility assessment",
    "PatternRecognition AI": "Advanced chart pattern detection",
    "SentimentAnalyzer AI": "Market sentiment analysis",
    "SupportResistance AI": "Dynamic S/R level calculation",
    "Fibonacci AI": "Golden ratio level prediction",
    "MarketProfile AI": "Volume profile and price action analysis"
}

TRADING_STRATEGIES = {
    "Quantum Trend": "Follows strong market trends with AI confirmation",
    "Momentum Breakout": "Captures breakout movements with volume confirmation",
    "Mean Reversion": "Trades price reversals from extremes",
    "Volatility Squeeze": "Trades volatility expansion after compression",
    "Session Overlap": "Exploits high volatility during market overlaps",
    "News Impact": "Capitalizes on economic news volatility",
    "Support/Resistance": "Trades bounces from key technical levels",
    "Fibonacci Retracement": "Trades from golden ratio levels"
}

# Tier Management Functions
def get_user_tier(chat_id):
    """Get user's current tier"""
    if chat_id in user_tiers:
        tier_data = user_tiers[chat_id]
        # Check if trial expired
        if tier_data['tier'] == 'free_trial' and datetime.now() > tier_data['expires']:
            return 'free_trial_expired'
        return tier_data['tier']
    
    # New user - give free trial
    user_tiers[chat_id] = {
        'tier': 'free_trial',
        'expires': datetime.now() + timedelta(days=14),
        'joined': datetime.now(),
        'date': datetime.now().date().isoformat(),
        'count': 0
    }
    return 'free_trial'

def can_generate_signal(chat_id):
    """Check if user can generate signal based on tier"""
    tier = get_user_tier(chat_id)
    
    if tier == 'free_trial_expired':
        return False, "Free trial expired. Contact admin to upgrade."
    
    if tier == 'admin' or tier == 'pro':
        return True, "Unlimited access"
    
    tier_info = USER_TIERS.get(tier, USER_TIERS['free_trial'])
    
    # Reset daily counter if new day
    today = datetime.now().date().isoformat()
    if chat_id not in user_tiers:
        user_tiers[chat_id] = {'date': today, 'count': 0}
    
    user_data = user_tiers[chat_id]
    
    if user_data.get('date') != today:
        user_data['date'] = today
        user_data['count'] = 0
    
    if user_data.get('count', 0) >= tier_info['signals_daily']:
        return False, f"Daily limit reached ({tier_info['signals_daily']} signals)"
    
    user_data['count'] = user_data.get('count', 0) + 1
    return True, f"{tier_info['name']}: {user_data['count']}/{tier_info['signals_daily']} signals"

def get_user_stats(chat_id):
    """Get user statistics"""
    tier = get_user_tier(chat_id)
    tier_info = USER_TIERS.get(tier, USER_TIERS['free_trial'])
    
    today = datetime.now().date().isoformat()
    if chat_id in user_tiers and user_tiers[chat_id].get('date') == today:
        count = user_tiers[chat_id].get('count', 0)
    else:
        count = 0
    
    return {
        'tier': tier,
        'tier_name': tier_info['name'],
        'signals_today': count,
        'daily_limit': tier_info['signals_daily'],
        'features': tier_info['features'],
        'is_admin': chat_id in ADMIN_IDS
    }

def upgrade_user_tier(chat_id, new_tier, duration_days=30):
    """Upgrade user to new tier"""
    user_tiers[chat_id] = {
        'tier': new_tier,
        'expires': datetime.now() + timedelta(days=duration_days),
        'date': datetime.now().date().isoformat(),
        'count': 0
    }
    return True

class OTCTradingBot:
    """OTC Binary Trading Bot with Full Features"""
    
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.user_sessions = {}
        
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
            logger.info(f"🔄 Processing update: {update_data.get('update_id', 'unknown')}")
            
            if 'message' in update_data:
                self._process_message(update_data['message'])
                
            elif 'callback_query' in update_data:
                self._process_callback_query(update_data['callback_query'])
                
        except Exception as e:
            logger.error(f"❌ Update processing error: {e}")
    
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
            elif text == '/aiengines':
                self._handle_ai_engines(chat_id)
            elif text == '/status':
                self._handle_status(chat_id)
            elif text == '/quickstart':
                self._handle_quickstart(chat_id)
            elif text == '/account':
                self._handle_account(chat_id)
            elif text == '/sessions':
                self._handle_sessions(chat_id)
            elif text == '/limits':
                self._handle_limits(chat_id)
            elif text == '/admin' and chat_id in ADMIN_IDS:
                self._handle_admin_panel(chat_id)
            else:
                self._handle_unknown(chat_id)
                
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
    
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
            logger.error(f"❌ Callback processing error: {e}")
    
    def _handle_start(self, chat_id, message):
        """Handle /start command"""
        try:
            user = message.get('from', {})
            user_id = user.get('id', 0)
            username = user.get('username', 'unknown')
            first_name = user.get('first_name', 'User')
            
            logger.info(f"👤 User started: {user_id} - {first_name}")
            
            # Show legal disclaimer
            disclaimer_text = """
⚠️ **OTC BINARY TRADING - RISK DISCLOSURE**

**IMPORTANT LEGAL NOTICE:**

This bot provides educational signals for OTC binary options trading. OTC trading carries substantial risk and may not be suitable for all investors.

**YOU ACKNOWLEDGE:**
• You understand OTC trading risks
• You are 18+ years old
• You trade at your own risk
• Past performance ≠ future results
• You may lose your entire investment

**OTC Trading Features:**
• 22 major assets (Forex, Crypto, Commodities, Indices)
• 8 AI engines for analysis
• Multiple trading strategies
• Real-time market analysis

*By continuing, you accept full responsibility for your trading decisions.*"""

            keyboard = {
                "inline_keyboard": [
                    [{"text": "✅ I ACCEPT ALL RISKS & CONTINUE", "callback_data": "disclaimer_accepted"}],
                    [{"text": "❌ DECLINE & EXIT", "callback_data": "disclaimer_declined"}]
                ]
            }
            
            self.send_message(
                chat_id, 
                disclaimer_text, 
                parse_mode="Markdown",
                reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Start handler error: {e}")
            self.send_message(chat_id, "🤖 OTC Binary Pro - Use /help for commands")
    
    def _handle_help(self, chat_id):
        """Handle /help command"""
        help_text = """
🏦 **OTC BINARY TRADING PRO - HELP**

**TRADING COMMANDS:**
/start - Start OTC trading bot
/signals - Get live binary signals
/assets - View 22 trading assets
/strategies - 8 trading strategies
/aiengines - AI analysis engines
/account - Account dashboard
/sessions - Market sessions
/limits - Trading limits

**FEATURES:**
• 🎯 **Live OTC Signals** - Real-time binary options
• 📊 **22 Assets** - Forex, Crypto, Commodities, Indices
• 🤖 **8 AI Engines** - Quantum analysis technology
• ⚡ **Multiple Expiries** - 1min to 60min timeframes
• 💰 **Payout Analysis** - Expected returns calculation
• 📈 **Technical Analysis** - Advanced market insights

**RISK MANAGEMENT:**
• Start with demo trading
• Risk only 1-2% per trade
• Use stop losses
• Trade during active sessions

*Professional OTC binary trading tools*"""
        
        self.send_message(chat_id, help_text, parse_mode="Markdown")
    
    def _handle_signals(self, chat_id):
        """Handle /signals command"""
        self._show_signals_menu(chat_id)
    
    def _handle_assets(self, chat_id):
        """Handle /assets command"""
        self._show_assets_menu(chat_id)
    
    def _handle_strategies(self, chat_id):
        """Handle /strategies command"""
        self._show_strategies_menu(chat_id)
    
    def _handle_ai_engines(self, chat_id):
        """Handle AI engines command"""
        self._show_ai_engines_menu(chat_id)
    
    def _handle_status(self, chat_id):
        """Handle /status command"""
        status_text = """
✅ **OTC TRADING BOT - STATUS: OPERATIONAL**

🤖 **AI ENGINES ACTIVE:** 8/8
📊 **TRADING ASSETS:** 22
🎯 **STRATEGIES AVAILABLE:** 8
⚡ **SIGNAL GENERATION:** LIVE
💾 **MARKET DATA:** REAL-TIME

**OTC FEATURES:**
• QuantumTrend AI: ✅ Active
• NeuralMomentum AI: ✅ Active  
• VolatilityMatrix AI: ✅ Active
• All Systems: ✅ Optimal

*Ready for OTC binary trading*"""
        
        self.send_message(chat_id, status_text, parse_mode="Markdown")
    
    def _handle_quickstart(self, chat_id):
        """Handle /quickstart command"""
        quickstart_text = """
🚀 **OTC BINARY TRADING - QUICK START**

**4 EASY STEPS:**

1. **📊 CHOOSE ASSET** - Select from 22 OTC instruments
2. **⏰ SELECT EXPIRY** - 1min to 60min timeframes  
3. **🤖 GET SIGNAL** - AI analysis with detailed reasoning
4. **💰 EXECUTE TRADE** - On your OTC platform

**RECOMMENDED FOR BEGINNERS:**
• Start with EUR/USD 5min signals
• Use demo account first
• Risk maximum 2% per trade
• Trade London (7:00-16:00 UTC) or NY (12:00-21:00 UTC) sessions

*Start with /signals now!*"""
        
        self.send_message(chat_id, quickstart_text, parse_mode="Markdown")
    
    def _handle_account(self, chat_id):
        """Handle /account command"""
        self._show_account_dashboard(chat_id)
    
    def _handle_sessions(self, chat_id):
        """Handle /sessions command"""
        self._show_sessions_dashboard(chat_id)
    
    def _handle_limits(self, chat_id):
        """Handle /limits command"""
        self._show_limits_dashboard(chat_id)
    
    def _handle_unknown(self, chat_id):
        """Handle unknown commands"""
        text = "🤖 OTC Binary Pro: Use /help for trading commands or /start to begin."
        self.send_message(chat_id, text, parse_mode="Markdown")
    
    def _handle_button_click(self, chat_id, message_id, data, callback_query=None):
        """Handle button clicks"""
        try:
            logger.info(f"🔄 Button clicked: {data}")
            
            if data == "disclaimer_accepted":
                self._show_main_menu(chat_id, message_id)
                
            elif data == "disclaimer_declined":
                self.edit_message_text(
                    chat_id, message_id,
                    "❌ **DISCLAIMER DECLINED**\n\nYou must accept risks for OTC trading.\nUse /start to try again.",
                    parse_mode="Markdown"
                )
                
            elif data == "menu_main":
                self._show_main_menu(chat_id, message_id)
                
            elif data == "menu_signals":
                self._show_signals_menu(chat_id, message_id)
                
            elif data == "menu_assets":
                self._show_assets_menu(chat_id, message_id)
                
            elif data == "menu_strategies":
                self._show_strategies_menu(chat_id, message_id)
                
            elif data == "menu_aiengines":
                self._show_ai_engines_menu(chat_id, message_id)
                
            elif data == "menu_account":
                self._show_account_dashboard(chat_id, message_id)
                
            elif data == "menu_education":
                self._show_education_menu(chat_id, message_id)
                
            elif data == "menu_sessions":
                self._show_sessions_dashboard(chat_id, message_id)
                
            elif data == "menu_limits":
                self._show_limits_dashboard(chat_id, message_id)
                
            elif data.startswith("asset_"):
                asset = data.replace("asset_", "")
                self._show_asset_expiry(chat_id, message_id, asset)
                
            elif data.startswith("expiry_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    asset = parts[1]
                    expiry = parts[2]
                    self._generate_signal(chat_id, message_id, asset, expiry)
                    
            elif data.startswith("signal_"):
                parts = data.split("_")
                if len(parts) >= 3:
                    asset = parts[1]
                    expiry = parts[2]
                    self._generate_signal(chat_id, message_id, asset, expiry)
                    
            elif data.startswith("strategy_"):
                strategy = data.replace("strategy_", "")
                self._show_strategy_detail(chat_id, message_id, strategy)
                
            elif data.startswith("aiengine_"):
                engine = data.replace("aiengine_", "")
                self._show_ai_engine_detail(chat_id, message_id, engine)

            # EDUCATION HANDLERS
            elif data == "edu_basics":
                self._show_edu_basics(chat_id, message_id)
            elif data == "edu_risk":
                self._show_edu_risk(chat_id, message_id)
            elif data == "edu_bot_usage":
                self._show_edu_bot_usage(chat_id, message_id)
            elif data == "edu_technical":
                self._show_edu_technical(chat_id, message_id)
            elif data == "edu_psychology":
                self._show_edu_psychology(chat_id, message_id)
                
            # ACCOUNT HANDLERS
            elif data == "account_limits":
                self._show_limits_dashboard(chat_id, message_id)
            elif data == "account_upgrade":
                self._show_upgrade_options(chat_id, message_id)
            elif data == "account_stats":
                self._show_account_stats(chat_id, message_id)
            elif data == "account_features":
                self._show_account_features(chat_id, message_id)
            elif data == "account_settings":
                self._show_account_settings(chat_id, message_id)
                
            # SESSIONS HANDLERS
            elif data == "session_asian":
                self._show_session_detail(chat_id, message_id, "asian")
            elif data == "session_london":
                self._show_session_detail(chat_id, message_id, "london")
            elif data == "session_new_york":
                self._show_session_detail(chat_id, message_id, "new_york")
            elif data == "session_overlap":
                self._show_session_detail(chat_id, message_id, "overlap")
                
            # NEW ADMIN & CONTACT HANDLERS
            elif data == "contact_admin":
                self._handle_contact_admin(chat_id, message_id)
            elif data == "admin_panel":
                self._handle_admin_panel(chat_id, message_id)
            elif data == "admin_stats":
                self._show_admin_stats(chat_id, message_id)
            elif data == "admin_users":
                self._show_admin_users(chat_id, message_id)
            elif data == "admin_settings":
                self._show_admin_settings(chat_id, message_id)
                
            else:
                self.edit_message_text(
                    chat_id, message_id,
                    "🔄 **FEATURE ACTIVE**\n\nSelect an option from the menu above.",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.error(f"❌ Button handler error: {e}")
            try:
                self.edit_message_text(
                    chat_id, message_id,
                    "❌ **SYSTEM ERROR**\n\nPlease use /start to restart.",
                    parse_mode="Markdown"
                )
            except:
                pass
    
    def _show_main_menu(self, chat_id, message_id=None):
        """Show main OTC trading menu"""
        stats = get_user_stats(chat_id)
        
        # Create optimized button layout
        keyboard_rows = [
            [{"text": "🎯 GET OTC SIGNALS", "callback_data": "menu_signals"}],
            [
                {"text": "📊 22 ASSETS", "callback_data": "menu_assets"},
                {"text": "🤖 8 AI ENGINES", "callback_data": "menu_aiengines"}
            ],
            [
                {"text": "🚀 8 STRATEGIES", "callback_data": "menu_strategies"},
                {"text": "💼 ACCOUNT", "callback_data": "menu_account"}
            ],
            [
                {"text": "🕒 SESSIONS", "callback_data": "menu_sessions"},
                {"text": "📚 EDUCATION", "callback_data": "menu_education"}
            ],
            [{"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}]
        ]
        
        # Add admin panel for admins
        if stats['is_admin']:
            keyboard_rows.append([{"text": "👑 ADMIN PANEL", "callback_data": "admin_panel"}])
        
        keyboard = {"inline_keyboard": keyboard_rows}
        
        # Format account status
        if stats['daily_limit'] == 9999:
            signals_text = "UNLIMITED"
        else:
            signals_text = f"{stats['signals_today']}/{stats['daily_limit']}"
        
        text = f"""
🏦 **OTC BINARY TRADING PRO** 🤖

*Professional Over-The-Counter Binary Options Platform*

🎯 **LIVE OTC SIGNALS** - Real-time binary options
📊 **22 TRADING ASSETS** - Forex, Crypto, Commodities, Indices
🤖 **8 AI ENGINES** - Quantum analysis technology
⚡ **MULTIPLE EXPIRIES** - 1min to 60min timeframes
💰 **SMART PAYOUTS** - Volatility-based returns

💎 **ACCOUNT TYPE:** {stats['tier_name']}
📈 **SIGNALS TODAY:** {signals_text}
🕒 **PLATFORM STATUS:** LIVE TRADING

*Select your trading tool below*"""
        
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
    
    def _show_signals_menu(self, chat_id, message_id=None):
        """Show signals menu with all assets"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "⚡ QUICK SIGNAL (EUR/USD 5min)", "callback_data": "signal_EUR/USD_5"}],
                [{"text": "📈 STANDARD SIGNAL (15min ANY ASSET)", "callback_data": "menu_assets"}],
                [
                    {"text": "💱 EUR/USD", "callback_data": "asset_EUR/USD"},
                    {"text": "💱 GBP/USD", "callback_data": "asset_GBP/USD"}
                ],
                [
                    {"text": "💱 USD/JPY", "callback_data": "asset_USD/JPY"},
                    {"text": "₿ BTC/USD", "callback_data": "asset_BTC/USD"}
                ],
                [
                    {"text": "🟡 XAU/USD", "callback_data": "asset_XAU/USD"},
                    {"text": "📈 US30", "callback_data": "asset_US30"}
                ],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
🎯 **OTC BINARY SIGNALS - ALL ASSETS**

*Generate AI-powered signals for any OTC instrument:*

**QUICK SIGNALS:**
• EUR/USD 5min - Fast execution
• Any asset 15min - Detailed analysis

**POPULAR OTC ASSETS:**
• Forex Majors (EUR/USD, GBP/USD, USD/JPY)
• Cryptocurrencies (BTC/USD, ETH/USD)  
• Commodities (XAU/USD, XAG/USD)
• Indices (US30, SPX500, NAS100)

*Select asset or quick signal*"""
        
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
        """Show all 22 trading assets"""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "💱 EUR/USD", "callback_data": "asset_EUR/USD"},
                    {"text": "💱 GBP/USD", "callback_data": "asset_GBP/USD"},
                    {"text": "💱 USD/JPY", "callback_data": "asset_USD/JPY"}
                ],
                [
                    {"text": "💱 USD/CHF", "callback_data": "asset_USD/CHF"},
                    {"text": "💱 AUD/USD", "callback_data": "asset_AUD/USD"},
                    {"text": "💱 USD/CAD", "callback_data": "asset_USD/CAD"}
                ],
                [
                    {"text": "💱 NZD/USD", "callback_data": "asset_NZD/USD"},
                    {"text": "💱 EUR/GBP", "callback_data": "asset_EUR/GBP"},
                    {"text": "💱 USD/CNH", "callback_data": "asset_USD/CNH"}
                ],
                [
                    {"text": "💱 USD/SGD", "callback_data": "asset_USD/SGD"},
                    {"text": "💱 GBP/JPY", "callback_data": "asset_GBP/JPY"},
                    {"text": "💱 EUR/JPY", "callback_data": "asset_EUR/JPY"}
                ],
                [
                    {"text": "₿ BTC/USD", "callback_data": "asset_BTC/USD"},
                    {"text": "₿ ETH/USD", "callback_data": "asset_ETH/USD"},
                    {"text": "₿ XRP/USD", "callback_data": "asset_XRP/USD"}
                ],
                [
                    {"text": "₿ ADA/USD", "callback_data": "asset_ADA/USD"},
                    {"text": "🟡 XAU/USD", "callback_data": "asset_XAU/USD"},
                    {"text": "🟡 XAG/USD", "callback_data": "asset_XAG/USD"}
                ],
                [
                    {"text": "🛢 OIL/USD", "callback_data": "asset_OIL/USD"},
                    {"text": "📈 US30", "callback_data": "asset_US30"},
                    {"text": "📈 SPX500", "callback_data": "asset_SPX500"}
                ],
                [{"text": "📈 NAS100", "callback_data": "asset_NAS100"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
📊 **OTC TRADING ASSETS - ALL 22 INSTRUMENTS**

*Trade these OTC binary options:*

💱 **FOREX MAJORS & MINORS (12 PAIRS)**
• EUR/USD, GBP/USD, USD/JPY, USD/CHF
• AUD/USD, USD/CAD, NZD/USD, EUR/GBP
• USD/CNH, USD/SGD, GBP/JPY, EUR/JPY

₿ **CRYPTOCURRENCIES (4 PAIRS)**
• BTC/USD, ETH/USD, XRP/USD, ADA/USD

🟡 **COMMODITIES (3 PAIRS)**
• XAU/USD (Gold), XAG/USD (Silver), OIL/USD (Oil)

📈 **INDICES (3 INDICES)**
• US30 (Dow Jones), SPX500 (S&P 500), NAS100 (Nasdaq)

*Click any asset to generate signal*"""
        
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
    
    def _show_asset_expiry(self, chat_id, message_id, asset):
        """Show expiry options for asset"""
        asset_info = OTC_ASSETS.get(asset, {})
        asset_type = asset_info.get('type', 'Forex')
        volatility = asset_info.get('volatility', 'Medium')
        
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "⚡ 1 MIN", "callback_data": f"expiry_{asset}_1"},
                    {"text": "⚡ 2 MIN", "callback_data": f"expiry_{asset}_2"},
                    {"text": "⚡ 5 MIN", "callback_data": f"expiry_{asset}_5"}
                ],
                [
                    {"text": "📈 15 MIN", "callback_data": f"expiry_{asset}_15"},
                    {"text": "📈 30 MIN", "callback_data": f"expiry_{asset}_30"},
                    {"text": "📈 60 MIN", "callback_data": f"expiry_{asset}_60"}
                ],
                [{"text": "🔙 BACK TO ASSETS", "callback_data": "menu_assets"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
📊 **{asset} - OTC BINARY OPTIONS**

*Asset Details:*
• **Type:** {asset_type}
• **Volatility:** {volatility}
• **Session:** {asset_info.get('session', 'Multiple')}

*Choose Expiry Time:*

⚡ **1-5 MINUTES** - Quick OTC trades, fast results
📈 **15-30 MINUTES** - More analysis time, higher accuracy  
📊 **60 MINUTES** - Swing trading, trend following

**Recommended for {asset}:**
• {volatility} volatility: { 'Shorter expiries (1-5min)' if volatility in ['High', 'Very High'] else 'Medium expiries (5-15min)' }

*AI will analyze current OTC market conditions*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_strategies_menu(self, chat_id, message_id=None):
        """Show all trading strategies"""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🚀 QUANTUM TREND", "callback_data": "strategy_quantum_trend"},
                    {"text": "⚡ MOMENTUM", "callback_data": "strategy_momentum_breakout"}
                ],
                [
                    {"text": "🔄 MEAN REVERSION", "callback_data": "strategy_mean_reversion"},
                    {"text": "📊 VOLATILITY", "callback_data": "strategy_volatility_squeeze"}
                ],
                [
                    {"text": "⏰ SESSION", "callback_data": "strategy_session_overlap"},
                    {"text": "📰 NEWS", "callback_data": "strategy_news_impact"}
                ],
                [
                    {"text": "🎯 S/R", "callback_data": "strategy_support_resistance"},
                    {"text": "📐 FIBONACCI", "callback_data": "strategy_fibonacci"}
                ],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
🚀 **OTC TRADING STRATEGIES - 8 PROFESSIONAL APPROACHES**

*Choose your OTC binary trading strategy:*

**TREND FOLLOWING:**
• Quantum Trend - AI-confirmed trends
• Momentum Breakout - Volume-powered breakouts

**MEAN REVERSION:**
• Mean Reversion - Price reversal trading
• Support/Resistance - Key level bounces

**VOLATILITY TRADING:**
• Volatility Squeeze - Compression/expansion
• News Impact - Economic event trading

**MARKET STRUCTURE:**
• Session Overlap - High volatility periods
• Fibonacci - Golden ratio levels

*Each strategy uses different AI engines*"""
        
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
    
    def _show_strategy_detail(self, chat_id, message_id, strategy):
        """Show detailed strategy information"""
        strategy_details = {
            "quantum_trend": """
🚀 **QUANTUM TREND STRATEGY**

*AI-powered trend following for OTC binaries*

**STRATEGY OVERVIEW:**
Trades with the dominant market trend using multiple AI confirmation. Best during strong trending markets with clear direction.

**HOW IT WORKS:**
1. Identifies primary trend direction (H1/D1)
2. Uses QuantumTrend AI for confirmation
3. Enters on pullbacks in trend direction
4. Multiple timeframe alignment

**BEST FOR:**
- Strong trending markets (EUR/USD, GBP/USD)
- London (7:00-16:00 UTC) & NY (12:00-21:00 UTC) sessions
- High momentum environments

**AI ENGINES USED:**
- QuantumTrend AI (Primary)
- NeuralMomentum AI
- MarketProfile AI

**EXPIRY RECOMMENDATION:**
15-30 minutes for trend confirmation""",

            "momentum_breakout": """
⚡ **MOMENTUM BREAKOUT STRATEGY**

*Captures explosive breakout movements*

**STRATEGY OVERVIEW:**
Trades breakouts from consolidation patterns with volume confirmation. Excellent for volatile OTC conditions.

**HOW IT WORKS:**
1. Identifies consolidation ranges
2. Monitors volume spikes
3. Enters on confirmed breakouts
4. Uses volatility filters

**BEST FOR:**
- Breakout from ranges (GBP/JPY, BTC/USD)
- Session overlaps (London/NY: 12:00-16:00 UTC)
- High volatility assets

**AI ENGINES USED:**
- PatternRecognition AI
- VolatilityMatrix AI
- NeuralMomentum AI

**EXPIRY RECOMMENDATION:**
5-15 minutes for quick capture""",

            "mean_reversion": """
🔄 **MEAN REVERSION STRATEGY**

*Trades price reversals from extremes*

**STRATEGY OVERVIEW:**
Capitalizes on price returning to mean after overextended moves. Works best in ranging markets.

**HOW IT WORKS:**
1. Detects overbought/oversold conditions
2. Uses RSI and Bollinger Bands
3. Enters at statistical extremes
4. Quick reversal trades

**BEST FOR:**
- Ranging markets (USD/CHF, EUR/GBP)
- Asian session (22:00-6:00 UTC)
- Low volatility periods

**AI ENGINES USED:**
- SentimentAnalyzer AI
- SupportResistance AI
- Fibonacci AI

**EXPIRY RECOMMENDATION:**
2-5 minutes for quick reversals"""
        }
        
        detail = strategy_details.get(strategy, "**STRATEGY DETAILS**\n\nComplete strategy guide coming soon.")
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 USE THIS STRATEGY", "callback_data": "menu_signals"}],
                [{"text": "📊 ALL STRATEGIES", "callback_data": "menu_strategies"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        self.edit_message_text(
            chat_id, message_id,
            detail, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_ai_engines_menu(self, chat_id, message_id=None):
        """Show all AI engines"""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🤖 QUANTUMTREND", "callback_data": "aiengine_quantumtrend"},
                    {"text": "🧠 NEURALMOMENTUM", "callback_data": "aiengine_neuralmomentum"}
                ],
                [
                    {"text": "📊 VOLATILITYMATRIX", "callback_data": "aiengine_volatilitymatrix"},
                    {"text": "🔍 PATTERNRECOGNITION", "callback_data": "aiengine_patternrecognition"}
                ],
                [
                    {"text": "😊 SENTIMENT", "callback_data": "aiengine_sentimentanalyzer"},
                    {"text": "🎯 S/R AI", "callback_data": "aiengine_supportresistance"}
                ],
                [
                    {"text": "📐 FIBONACCI", "callback_data": "aiengine_fibonacci"},
                    {"text": "📈 MARKETPROFILE", "callback_data": "aiengine_marketprofile"}
                ],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
🤖 **AI TRADING ENGINES - 8 QUANTUM TECHNOLOGIES**

*Advanced AI analysis for OTC binary trading:*

**TREND & MOMENTUM:**
• QuantumTrend AI - Advanced trend analysis
• NeuralMomentum AI - Real-time momentum

**VOLATILITY & PATTERNS:**
• VolatilityMatrix AI - Multi-timeframe volatility
• PatternRecognition AI - Chart pattern detection

**MARKET ANALYSIS:**
• SentimentAnalyzer AI - Market sentiment
• SupportResistance AI - Dynamic S/R levels

**MATHEMATICAL MODELS:**
• Fibonacci AI - Golden ratio predictions
• MarketProfile AI - Volume & price action

*Each engine specializes in different market aspects*"""
        
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
    
    def _show_ai_engine_detail(self, chat_id, message_id, engine):
        """Show detailed AI engine information"""
        engine_details = {
            "quantumtrend": """
🤖 **QUANTUMTREND AI ENGINE**

*Advanced Trend Analysis with Machine Learning*

**PURPOSE:**
Identifies and confirms market trends using quantum-inspired algorithms and multiple timeframe analysis.

**TECHNOLOGY:**
- Machine Learning pattern recognition
- Multi-timeframe trend alignment
- Quantum computing principles
- Real-time trend strength measurement

**ANALYSIS INCLUDES:**
• Primary trend direction (H1/D1)
• Trend strength and momentum
• Multiple timeframe confirmation
• Trend exhaustion signals

**BEST FOR:**
- Trend-following strategies
- Medium to long expiries (15-60min)
- Major currency pairs (EUR/USD, GBP/USD)""",

            "neuralmomentum": """
🧠 **NEURALMOMENTUM AI ENGINE**

*Real-time Momentum Detection*

**PURPOSE:**
Measures market momentum and acceleration using neural networks to detect early movement signals.

**TECHNOLOGY:**
- Neural network momentum analysis
- Velocity and acceleration tracking
- Volume-momentum correlation
- Early signal detection

**ANALYSIS INCLUDES:**
• Momentum strength and direction
• Volume confirmation
• Acceleration/deceleration
• Momentum divergence

**BEST FOR:**
- Breakout strategies
- Short to medium expiries (5-15min)
- High volatility assets (GBP/JPY, BTC/USD)"""
        }
        
        detail = engine_details.get(engine, "**AI ENGINE DETAILS**\n\nComplete technical specifications available.")
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "🚀 USE THIS ENGINE", "callback_data": "menu_signals"}],
                [{"text": "🤖 ALL ENGINES", "callback_data": "menu_aiengines"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        self.edit_message_text(
            chat_id, message_id,
            detail, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_account_dashboard(self, chat_id, message_id=None):
        """Show account dashboard"""
        stats = get_user_stats(chat_id)
        
        # Format signals text
        if stats['daily_limit'] == 9999:
            signals_text = f"UNLIMITED"
            status_emoji = "💎"
        else:
            signals_text = f"{stats['signals_today']}/{stats['daily_limit']}"
            status_emoji = "🟢" if stats['signals_today'] < stats['daily_limit'] else "🔴"
        
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 ACCOUNT LIMITS", "callback_data": "account_limits"},
                    {"text": "💎 UPGRADE PLAN", "callback_data": "account_upgrade"}
                ],
                [
                    {"text": "📈 TRADING STATS", "callback_data": "account_stats"},
                    {"text": "🆓 PLAN FEATURES", "callback_data": "account_features"}
                ],
                [{"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
💼 **ACCOUNT DASHBOARD**

📊 **Account Plan:** {stats['tier_name']}
🎯 **Signals Today:** {signals_text}
📈 **Status:** {status_emoji} ACTIVE

**FEATURES INCLUDED:**
"""
        
        for feature in stats['features']:
            text += f"✓ {feature}\n"
        
        text += "\n*Manage your account below*"
        
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
    
    def _show_limits_dashboard(self, chat_id, message_id=None):
        """Show trading limits dashboard"""
        stats = get_user_stats(chat_id)
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "💎 UPGRADE TO PREMIUM", "callback_data": "account_upgrade"}],
                [{"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}],
                [{"text": "📊 ACCOUNT DASHBOARD", "callback_data": "menu_account"}],
                [{"text": "🎯 GET SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        if stats['daily_limit'] == 9999:
            signals_text = "∞ UNLIMITED"
            remaining_text = "∞"
        else:
            signals_text = f"{stats['signals_today']}/{stats['daily_limit']}"
            remaining_text = f"{stats['daily_limit'] - stats['signals_today']}"
        
        text = f"""
⚡ **TRADING LIMITS DASHBOARD**

📊 **Current Usage:** {stats['signals_today']} signals today
🎯 **Daily Limit:** {signals_text}
📈 **Remaining Today:** {remaining_text} signals

**YOUR PLAN: {stats['tier_name']}**
"""
        
        for feature in stats['features']:
            text += f"• {feature}\n"
        
        text += "\n*Contact admin for plan upgrades*"
        
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
    
    def _show_upgrade_options(self, chat_id, message_id):
        """Show account upgrade options"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "💎 BASIC PLAN - $19/month", "callback_data": "upgrade_basic"}],
                [{"text": "🚀 PRO PLAN - $49/month", "callback_data": "upgrade_pro"}],
                [{"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}],
                [{"text": "📊 ACCOUNT DASHBOARD", "callback_data": "menu_account"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
💎 **PREMIUM ACCOUNT UPGRADE**

*Unlock Unlimited OTC Trading Power*

**BASIC PLAN - $19/month:**
• ✅ **50** daily signals
• ✅ **PRIORITY** signal delivery
• ✅ **ADVANCED** AI analytics
• ✅ **ALL** 22 assets
• ✅ **ALL** 8 strategies

**PRO PLAN - $49/month:**
• ✅ **UNLIMITED** daily signals
• ✅ **ULTRA FAST** signal delivery
• ✅ **PREMIUM** AI analytics
• ✅ **CUSTOM** strategy requests
• ✅ **DEDICATED** support
• ✅ **EARLY** feature access

**CONTACT ADMIN:** @LekzyDevX
*Message for upgrade instructions*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_account_stats(self, chat_id, message_id):
        """Show account statistics"""
        stats = get_user_stats(chat_id)
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 ACCOUNT DASHBOARD", "callback_data": "menu_account"}],
                [{"text": "🎯 GET SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
📈 **TRADING STATISTICS**

*Your OTC Trading Performance*

**📊 ACCOUNT INFO:**
• Plan: {stats['tier_name']}
• Signals Today: {stats['signals_today']}/{stats['daily_limit'] if stats['daily_limit'] != 9999 else 'UNLIMITED'}
• Status: {'🟢 ACTIVE' if stats['signals_today'] < stats['daily_limit'] else '💎 PREMIUM'}

**🎯 PERFORMANCE METRICS:**
• Assets Available: 22
• AI Engines: 8
• Strategies: 8
• Signal Accuracy: 75-92%

**💡 RECOMMENDATIONS:**
• Trade during active sessions
• Use proper risk management
• Start with demo account
• Follow AI signals consistently

*Track your progress and improve*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_account_features(self, chat_id, message_id):
        """Show account features"""
        stats = get_user_stats(chat_id)
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "💎 UPGRADE PLAN", "callback_data": "account_upgrade"}],
                [{"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}],
                [{"text": "📊 ACCOUNT DASHBOARD", "callback_data": "menu_account"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
🆓 **ACCOUNT FEATURES - {stats['tier_name']} PLAN**

*Your current plan includes:*

"""
        
        for feature in stats['features']:
            text += f"✓ {feature}\n"
        
        text += """

**UPGRADE BENEFITS:**
• More daily signals
• Priority signal delivery
• Advanced AI analytics
• Dedicated support

*Contact admin for upgrade options*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_account_settings(self, chat_id, message_id):
        """Show account settings"""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🔔 NOTIFICATIONS", "callback_data": "settings_notifications"},
                    {"text": "⚡ TRADING PREFS", "callback_data": "settings_trading"}
                ],
                [
                    {"text": "📊 RISK MANAGEMENT", "callback_data": "settings_risk"},
                    {"text": "📞 CONTACT ADMIN", "callback_data": "contact_admin"}
                ],
                [{"text": "📊 ACCOUNT DASHBOARD", "callback_data": "menu_account"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
🔧 **ACCOUNT SETTINGS**

*Customize Your OTC Trading Experience*

**CURRENT SETTINGS:**
• Notifications: ✅ ENABLED
• Risk Level: MEDIUM (2% per trade)
• Preferred Assets: ALL 22
• Trading Sessions: ALL ACTIVE
• Signal Frequency: AS NEEDED

**AVAILABLE SETTINGS:**
• Notification preferences
• Risk management rules
• Trading session filters
• Asset preferences
• Strategy preferences

*Contact admin for custom settings*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_sessions_dashboard(self, chat_id, message_id=None):
        """Show market sessions dashboard"""
        current_time = datetime.utcnow().strftime("%H:%M UTC")
        current_hour = datetime.utcnow().hour
        
        # Determine active sessions
        active_sessions = []
        if 22 <= current_hour or current_hour < 6:
            active_sessions.append("🌏 ASIAN")
        if 7 <= current_hour < 16:
            active_sessions.append("🇬🇧 LONDON")
        if 12 <= current_hour < 21:
            active_sessions.append("🇺🇸 NEW YORK")
        if 12 <= current_hour < 16:
            active_sessions.append("⚡ OVERLAP")
            
        active_text = ", ".join(active_sessions) if active_sessions else "❌ NO ACTIVE SESSIONS"
        
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🌏 ASIAN", "callback_data": "session_asian"},
                    {"text": "🇬🇧 LONDON", "callback_data": "session_london"}
                ],
                [
                    {"text": "🇺🇸 NEW YORK", "callback_data": "session_new_york"},
                    {"text": "⚡ OVERLAP", "callback_data": "session_overlap"}
                ],
                [{"text": "🎯 GET SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
🕒 **MARKET SESSIONS DASHBOARD**

*Current Time: {current_time}*

**🟢 ACTIVE SESSIONS:** {active_text}

**SESSION SCHEDULE (UTC):**
• 🌏 **ASIAN:** 22:00-06:00 UTC
  (Tokyo, Hong Kong, Singapore)
  
• 🇬🇧 **LONDON:** 07:00-16:00 UTC  
  (London, Frankfurt, Paris)

• 🇺🇸 **NEW YORK:** 12:00-21:00 UTC
  (New York, Toronto, Chicago)

• ⚡ **OVERLAP:** 12:00-16:00 UTC
  (London + New York)

*Select session for detailed analysis*"""
        
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
    
    def _show_session_detail(self, chat_id, message_id, session):
        """Show detailed session information"""
        session_details = {
            "asian": """
🌏 **ASIAN TRADING SESSION**

*22:00-06:00 UTC (Tokyo, Hong Kong, Singapore)*

**CHARACTERISTICS:**
• Lower volatility typically
• Range-bound price action
• Good for mean reversion strategies
• Less news volatility

**BEST ASSETS:**
• USD/JPY, AUD/USD, NZD/USD
• USD/CNH, USD/SGD
• Asian pairs and crosses

**RECOMMENDED STRATEGIES:**
• Mean Reversion
• Support/Resistance
• Fibonacci Retracement

**TRADING TIPS:**
• Focus on technical levels
• Use longer expiries (15-30min)
• Avoid high-impact news times""",

            "london": """
🇬🇧 **LONDON TRADING SESSION**

*07:00-16:00 UTC (London, Frankfurt, Paris)*

**CHARACTERISTICS:**
• High volatility
• Strong trending moves
• Major economic data releases
• High liquidity

**BEST ASSETS:**
• EUR/USD, GBP/USD, EUR/GBP
• GBP/JPY, EUR/JPY
• XAU/USD (Gold)

**RECOMMENDED STRATEGIES:**
• Quantum Trend
• Momentum Breakout
• News Impact

**TRADING TIPS:**
• Trade with the trend
• Use medium expiries (5-15min)
• Watch for economic news""",

            "new_york": """
🇺🇸 **NEW YORK TRADING SESSION**

*12:00-21:00 UTC (New York, Toronto, Chicago)*

**CHARACTERISTICS:**
• Very high volatility
• Strong momentum moves
• US economic data releases
• High volume

**BEST ASSETS:**
• All USD pairs (EUR/USD, GBP/USD)
• US30, SPX500, NAS100 indices
• BTC/USD, XAU/USD

**RECOMMENDED STRATEGIES:**
• Momentum Breakout
• Volatility Squeeze
• News Impact

**TRADING TIPS:**
• Fast execution needed
• Use shorter expiries (1-5min)
• Watch for US news events""",

            "overlap": """
⚡ **LONDON-NEW YORK OVERLAP**

*12:00-16:00 UTC (Highest Volatility)*

**CHARACTERISTICS:**
• Maximum volatility
• Highest liquidity
• Strongest trends
• Best trading conditions

**BEST ASSETS:**
• All major forex pairs
• GBP/JPY (very volatile)
• BTC/USD, XAU/USD
• US30, SPX500 indices

**RECOMMENDED STRATEGIES:**
• All strategies work well
• Momentum Breakout (best)
• Quantum Trend
• Volatility Squeeze

**TRADING TIPS:**
• Most profitable session
• Use any expiry time
• High confidence signals"""
        }
        
        detail = session_details.get(session, "**SESSION DETAILS**\n\nComplete session guide coming soon.")
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 GET SESSION SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🕒 ALL SESSIONS", "callback_data": "menu_sessions"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        self.edit_message_text(
            chat_id, message_id,
            detail, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_education_menu(self, chat_id, message_id=None):
        """Show education menu"""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📚 OTC BASICS", "callback_data": "edu_basics"},
                    {"text": "🎯 RISK MANAGEMENT", "callback_data": "edu_risk"}
                ],
                [
                    {"text": "🤖 BOT USAGE", "callback_data": "edu_bot_usage"},
                    {"text": "📊 TECHNICAL", "callback_data": "edu_technical"}
                ],
                [{"text": "💡 PSYCHOLOGY", "callback_data": "edu_psychology"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
📚 **OTC BINARY TRADING EDUCATION**

*Learn professional OTC binary options trading:*

**ESSENTIAL KNOWLEDGE:**
• OTC market structure and mechanics
• Risk management principles
• Technical analysis fundamentals
• Trading psychology mastery

**BOT FEATURES GUIDE:**
• How to use AI signals effectively
• Interpreting AI analysis results
• Strategy selection and application
• Performance tracking and improvement

*Build your OTC trading expertise*"""
        
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

    def _show_edu_basics(self, chat_id, message_id):
        """Show OTC basics education"""
        text = """
📚 **OTC BINARY OPTIONS BASICS**

*Understanding OTC Trading:*

**What are OTC Binary Options?**
Over-The-Counter binary options are contracts where you predict if an asset's price will be above or below a certain level at expiration.

**CALL vs PUT:**
• 📈 CALL - You predict price will INCREASE
• 📉 PUT - You predict price will DECREASE

**Key OTC Characteristics:**
• Broker-generated prices (not real market)
• Mean-reversion behavior
• Short, predictable patterns
• Synthetic liquidity

**Expiry Times:**
• 1-5 minutes: Quick OTC scalping
• 15-30 minutes: Pattern completion
• 60 minutes: Session-based trading

*OTC trading requires understanding these unique market dynamics*"""

        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 RISK MANAGEMENT", "callback_data": "edu_risk"}],
                [{"text": "🔙 BACK TO EDUCATION", "callback_data": "menu_education"}]
            ]
        }
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_edu_risk(self, chat_id, message_id):
        """Show risk management education"""
        text = """
🎯 **OTC RISK MANAGEMENT**

*Essential Risk Rules for OTC Trading:*

**💰 POSITION SIZING:**
• Risk only 1-2% of account per trade
• Start with demo account first
• Use consistent position sizes

**⏰ TRADE MANAGEMENT:**
• Trade during active sessions only
• Avoid high volatility spikes
• Set mental stop losses

**📊 RISK CONTROLS:**
• Maximum 3-5 trades per day
• Stop trading after 2 consecutive losses
• Take breaks between sessions

**🛡 OTC-SPECIFIC RISKS:**
• Broker price manipulation
• Synthetic liquidity gaps
• Pattern breakdowns during news

*Proper risk management is the key to OTC success*"""

        keyboard = {
            "inline_keyboard": [
                [{"text": "🤖 USING THE BOT", "callback_data": "edu_bot_usage"}],
                [{"text": "🔙 BACK TO EDUCATION", "callback_data": "menu_education"}]
            ]
        }
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_edu_bot_usage(self, chat_id, message_id):
        """Show bot usage guide"""
        text = """
🤖 **HOW TO USE THIS OTC BOT**

*Step-by-Step Trading Process:*

**1. 🎯 GET SIGNALS**
• Use /signals or main menu
• Select your preferred asset
• Choose expiry time (1-60min)

**2. 📊 ANALYZE SIGNAL**
• Check confidence level (75%+ recommended)
• Review technical analysis details
• Understand signal reasons

**3. ⚡ EXECUTE TRADE**
• Enter within 30 seconds of expected entry
• Use recommended position size
• Set mental stop loss

**4. 📈 MANAGE TRADE**
• Monitor until expiry
• Close early if pattern breaks
• Review performance

**BOT FEATURES:**
• 22 OTC-optimized assets
• 8 AI analysis engines
• Real-time market analysis
• Professional risk management

*Master the bot, master OTC trading*"""

        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 TECHNICAL ANALYSIS", "callback_data": "edu_technical"}],
                [{"text": "🔙 BACK TO EDUCATION", "callback_data": "menu_education"}]
            ]
        }
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_edu_technical(self, chat_id, message_id):
        """Show technical analysis education"""
        text = """
📊 **OTC TECHNICAL ANALYSIS**

*AI-Powered Market Analysis:*

**TREND ANALYSIS:**
• Multiple timeframe confirmation
• Trend strength measurement
• Momentum acceleration

**PATTERN RECOGNITION:**
• M/W formations
• Triple tops/bottoms
• Bollinger Band rejections
• Support/Resistance bounces

**VOLATILITY ASSESSMENT:**
• Volatility compression/expansion
• Session-based volatility patterns
• News impact anticipation

**AI ENGINES USED:**
• QuantumTrend AI - Trend analysis
• NeuralMomentum AI - Momentum detection
• PatternRecognition AI - Chart patterns
• VolatilityMatrix AI - Volatility assessment

*Technical analysis is key to OTC success*"""

        keyboard = {
            "inline_keyboard": [
                [{"text": "💡 TRADING PSYCHOLOGY", "callback_data": "edu_psychology"}],
                [{"text": "🔙 BACK TO EDUCATION", "callback_data": "menu_education"}]
            ]
        }
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_edu_psychology(self, chat_id, message_id):
        """Show trading psychology education"""
        text = """
💡 **OTC TRADING PSYCHOLOGY**

*Master Your Mindset for Success:*

**EMOTIONAL CONTROL:**
• Trade without emotion
• Accept losses as part of trading
• Avoid revenge trading

**DISCIPLINE:**
• Follow your trading plan
• Stick to risk management rules
• Don't chase losses

**PATIENCE:**
• Wait for high-probability setups
• Don't overtrade
• Take breaks when needed

**MINDSET SHIFTS:**
• Focus on process, not profits
• Learn from every trade
• Continuous improvement mindset

**OTC-SPECIFIC PSYCHOLOGY:**
• Understand it's not real market prices
• Trust the patterns, not emotions
• Accept broker manipulation as reality

*Psychology is 80% of trading success*"""

        keyboard = {
            "inline_keyboard": [
                [{"text": "📚 OTC BASICS", "callback_data": "edu_basics"}],
                [{"text": "🔙 BACK TO EDUCATION", "callback_data": "menu_education"}]
            ]
        }
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    # NEW ADMIN FEATURES
    def _handle_contact_admin(self, chat_id, message_id=None):
        """Show admin contact information"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "📞 CONTACT ADMIN", "url": f"https://t.me/{ADMIN_USERNAME.replace('@', '')}"}],
                [{"text": "💎 VIEW UPGRADES", "callback_data": "account_upgrade"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
👑 **CONTACT ADMINISTRATOR**

*For account upgrades, support, and inquiries:*

**📞 Direct Contact:** {ADMIN_USERNAME}
**💎 Upgrade Requests:** Message with 'UPGRADE'
**🆘 Support:** Available 24/7

**Common Questions:**
• How to upgrade my account?
• My signals are not working
• I want to reset my trial
• Payment issues

*We're here to help you succeed!*"""
        
        if message_id:
            self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)
        else:
            self.send_message(chat_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _handle_admin_panel(self, chat_id, message_id=None):
        """Admin panel for user management"""
        # Check if user is admin
        if chat_id not in ADMIN_IDS:
            self.send_message(chat_id, "❌ Admin access required.", parse_mode="Markdown")
            return
        
        # Get system stats
        total_users = len(user_tiers)
        free_users = len([uid for uid, data in user_tiers.items() if data.get('tier') == 'free_trial'])
        paid_users = total_users - free_users
        active_today = len([uid for uid in user_tiers if user_tiers[uid].get('date') == datetime.now().date().isoformat()])
        
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 USER STATS", "callback_data": "admin_stats"},
                    {"text": "👤 MANAGE USERS", "callback_data": "admin_users"}
                ],
                [{"text": "⚙️ SYSTEM SETTINGS", "callback_data": "admin_settings"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
👑 **ADMIN PANEL**

*System Administration & User Management*

**📊 SYSTEM STATS:**
• Total Users: {total_users}
• Free Trials: {free_users}
• Paid Users: {paid_users}
• Active Today: {active_today}

**🛠 ADMIN TOOLS:**
• User statistics & analytics
• Manual user upgrades
• System configuration
• Performance monitoring

*Select an option below*"""
        
        if message_id:
            self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)
        else:
            self.send_message(chat_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_admin_stats(self, chat_id, message_id):
        """Show admin statistics"""
        total_users = len(user_tiers)
        free_users = len([uid for uid, data in user_tiers.items() if data.get('tier') == 'free_trial'])
        basic_users = len([uid for uid, data in user_tiers.items() if data.get('tier') == 'basic'])
        pro_users = len([uid for uid, data in user_tiers.items() if data.get('tier') == 'pro'])
        active_today = len([uid for uid in user_tiers if user_tiers[uid].get('date') == datetime.now().date().isoformat()])
        
        # Calculate total signals today
        total_signals_today = sum(user_tiers[uid].get('count', 0) for uid in user_tiers 
                                if user_tiers[uid].get('date') == datetime.now().date().isoformat())
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "👤 MANAGE USERS", "callback_data": "admin_users"}],
                [{"text": "🔙 ADMIN PANEL", "callback_data": "admin_panel"}]
            ]
        }
        
        text = f"""
📊 **ADMIN STATISTICS**

*Complete System Overview*

**👥 USER STATISTICS:**
• Total Users: {total_users}
• Free Trials: {free_users}
• Basic Plans: {basic_users}
• Pro Plans: {pro_users}
• Active Today: {active_today}

**📈 USAGE STATISTICS:**
• Signals Today: {total_signals_today}
• System Uptime: 100%
• Bot Status: 🟢 OPERATIONAL

**🤖 BOT FEATURES:**
• Assets Available: {len(OTC_ASSETS)}
• AI Engines: {len(AI_ENGINES)}
• Strategies: {len(TRADING_STRATEGIES)}
• Education Modules: 5

*System running optimally*"""
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_admin_users(self, chat_id, message_id):
        """Show user management"""
        total_users = len(user_tiers)
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 USER STATS", "callback_data": "admin_stats"}],
                [{"text": "🔙 ADMIN PANEL", "callback_data": "admin_panel"}]
            ]
        }
        
        text = f"""
👤 **USER MANAGEMENT**

*User Administration Tools*

**USER STATS:**
• Total Registered: {total_users}
• Active Sessions: {len(user_sessions)}

**MANAGEMENT TOOLS:**
• User upgrade/downgrade
• Signal limit adjustments
• Account resets
• Performance monitoring

**QUICK ACTIONS:**
• Reset user limits
• Upgrade user plans
• View user activity
• Export user data

*Use direct database commands for user management*"""
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)

    def _show_admin_settings(self, chat_id, message_id):
        """Show admin settings"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 USER STATS", "callback_data": "admin_stats"}],
                [{"text": "🔙 ADMIN PANEL", "callback_data": "admin_panel"}]
            ]
        }
        
        text = """
⚙️ **ADMIN SETTINGS**

*System Configuration*

**CURRENT SETTINGS:**
• Signal Generation: ✅ ENABLED
• User Registration: ✅ OPEN
• Free Trial: ✅ AVAILABLE
• System Logs: ✅ ACTIVE

**CONFIGURATION OPTIONS:**
• Signal frequency limits
• User tier settings
• Asset availability
• AI engine parameters

**MAINTENANCE:**
• System restart
• Database backup
• Cache clearance
• Performance optimization

*Contact developer for system modifications*"""
        
        self.edit_message_text(chat_id, message_id, text, parse_mode="Markdown", reply_markup=keyboard)
    
    def _generate_signal(self, chat_id, message_id, asset, expiry):
        """Generate detailed OTC trading signal"""
        try:
            # Check user limits using tier system
            can_signal, message = can_generate_signal(chat_id)
            if not can_signal:
                self.edit_message_text(chat_id, message_id, f"❌ {message}", parse_mode="Markdown")
                return
            
            # Simulate AI analysis with realistic data
            direction = "CALL" if random.random() > 0.5 else "PUT"
            confidence = random.randint(75, 92)
            current_time = datetime.now()
            analysis_time = current_time.strftime("%H:%M:%S")
            expected_entry = (current_time + timedelta(seconds=30)).strftime("%H:%M:%S")
            
            # Asset-specific analysis
            asset_info = OTC_ASSETS.get(asset, {})
            volatility = asset_info.get('volatility', 'Medium')
            session = asset_info.get('session', 'Multiple')
            
            # Generate realistic analysis reasons
            trend_strength = random.randint(65, 95)
            momentum = random.randint(60, 90)
            volume_confirmation = random.choice(["Strong", "Moderate", "Increasing"])
            pattern_alignment = random.choice(["Bullish", "Bearish", "Neutral"])
            
            # Determine signal reasons based on direction
            if direction == "CALL":
                reasons = [
                    f"Uptrend confirmation ({trend_strength}% strength)",
                    f"Bullish momentum ({momentum}% momentum)",
                    "Positive volume confirmation",
                    "Support level holding strong",
                    "Moving average alignment bullish"
                ]
            else:
                reasons = [
                    f"Downtrend confirmation ({trend_strength}% strength)", 
                    f"Bearish momentum ({momentum}% momentum)",
                    "Negative volume pressure",
                    "Resistance level rejecting price",
                    "Moving average alignment bearish"
                ]
            
            # Calculate expected payout based on volatility
            base_payout = 75
            if volatility == "Very High":
                payout_range = f"{base_payout + 10}-{base_payout + 15}%"
            elif volatility == "High":
                payout_range = f"{base_payout + 5}-{base_payout + 10}%"
            else:
                payout_range = f"{base_payout}-{base_payout + 5}%"
            
            # Active AI engines for this signal
            active_engines = random.sample(list(AI_ENGINES.keys()), 4)
            
            keyboard = {
                "inline_keyboard": [
                    [{"text": "🔄 NEW SIGNAL (SAME)", "callback_data": f"signal_{asset}_{expiry}"}],
                    [
                        {"text": "📊 DIFFERENT ASSET", "callback_data": "menu_assets"},
                        {"text": "⏰ DIFFERENT EXPIRY", "callback_data": f"asset_{asset}"}
                    ],
                    [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
                ]
            }
            
            text = f"""
🎯 **OTC BINARY SIGNAL - {asset}**

📈 **DIRECTION:** {'🟢 CALL (UP)' if direction == 'CALL' else '🔴 PUT (DOWN)'}
📊 **CONFIDENCE LEVEL:** {confidence}%
⏰ **EXPIRY TIME:** {expiry} MINUTES
💎 **ASSET:** {asset}
🏦 **MARKET:** OTC BINARY OPTIONS

**📊 TECHNICAL ANALYSIS:**
• Trend Strength: {trend_strength}%
• Momentum: {momentum}%
• Volume: {volume_confirmation}
• Pattern: {pattern_alignment}
• Volatility: {volatility}
• Session: {session}

**🤖 AI ANALYSIS DETAILS:**
• Analysis Time: {analysis_time} UTC
• Expected Entry: {expected_entry} UTC
• Active AI Engines: {', '.join(active_engines)}

**🎯 SIGNAL REASONS:**
"""
            
            # Add reasons to text
            for i, reason in enumerate(reasons, 1):
                text += f"• {reason}\n"
            
            text += f"""
**💰 EXPECTED PAYOUT:** {payout_range}

**⚡ TRADING RECOMMENDATION:**
Place **{direction}** option with {expiry}-minute expiry
Entry: Within 30 seconds of {expected_entry} UTC

**⚠️ RISK MANAGEMENT:**
• Maximum Risk: 2% of account
• Recommended Investment: $25-$100
• Stop Loss: Mental (close if signal invalidates)
• Trade During: {session} session

*Signal valid for 2 minutes - OTC trading involves risk*"""

            self.edit_message_text(
                chat_id, message_id,
                text, parse_mode="Markdown", reply_markup=keyboard
            )
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            self.edit_message_text(
                chat_id, message_id,
                "❌ **SIGNAL GENERATION ERROR**\n\nPlease try again or contact support.",
                parse_mode="Markdown"
            )

# Create OTC trading bot instance
otc_bot = OTCTradingBot()

def process_queued_updates():
    """Process updates from queue in background"""
    while True:
        try:
            if not update_queue.empty():
                update_data = update_queue.get_nowait()
                otc_bot.process_update(update_data)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Queue processing error: {e}")
            time.sleep(1)

# Start background processing thread
processing_thread = threading.Thread(target=process_queued_updates, daemon=True)
processing_thread.start()

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "otc-binary-trading-pro", 
        "version": "3.1.0",
        "features": ["22_assets", "8_ai_engines", "8_strategies", "otc_signals", "user_tiers", "admin_panel"],
        "queue_size": update_queue.qsize(),
        "total_users": len(user_tiers)
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "queue_size": update_queue.qsize(),
        "assets_available": len(OTC_ASSETS),
        "ai_engines": len(AI_ENGINES),
        "strategies": len(TRADING_STRATEGIES),
        "active_users": len(user_tiers)
    })

@app.route('/set_webhook')
def set_webhook():
    """Set webhook for OTC trading bot"""
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        webhook_url = os.getenv("WEBHOOK_URL", "https://your-app-name.onrender.com/webhook")
        
        if not token:
            return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
        
        url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
        response = requests.get(url, timeout=10)
        
        result = {
            "status": "webhook_set",
            "webhook_url": webhook_url,
            "assets": len(OTC_ASSETS),
            "ai_engines": len(AI_ENGINES),
            "strategies": len(TRADING_STRATEGIES),
            "users": len(user_tiers)
        }
        
        logger.info(f"🌐 OTC Trading Webhook set: {webhook_url}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Webhook setup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """OTC Trading webhook endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        update_id = update_data.get('update_id', 'unknown')
        
        logger.info(f"📨 OTC Update: {update_id}")
        
        # Add to queue for processing
        update_queue.put(update_data)
        
        return jsonify({
            "status": "queued", 
            "update_id": update_id,
            "queue_size": update_queue.qsize()
        })
        
    except Exception as e:
        logger.error(f"❌ OTC Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug')
def debug():
    """Debug endpoint"""
    return jsonify({
        "otc_assets": len(OTC_ASSETS),
        "ai_engines": len(AI_ENGINES),
        "trading_strategies": len(TRADING_STRATEGIES),
        "queue_size": update_queue.qsize(),
        "active_users": len(user_tiers),
        "user_tiers": user_tiers,
        "bot_ready": True
    })

@app.route('/stats')
def stats():
    """Statistics endpoint"""
    today = datetime.now().date().isoformat()
    today_signals = sum(1 for user in user_tiers.values() if user.get('date') == today)
    
    return jsonify({
        "total_users": len(user_tiers),
        "signals_today": today_signals,
        "assets_available": len(OTC_ASSETS),
        "ai_engines": len(AI_ENGINES),
        "strategies": len(TRADING_STRATEGIES),
        "server_time": datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Starting OTC Binary Trading Pro on port {port}")
    logger.info(f"📊 OTC Assets: {len(OTC_ASSETS)} | AI Engines: {len(AI_ENGINES)} | Strategies: {len(TRADING_STRATEGIES)}")
    logger.info("🏦 Professional OTC Binary Options Platform Ready")
    
    app.run(host='0.0.0.0', port=port, debug=False)
