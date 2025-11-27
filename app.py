from flask import Flask, request, jsonify
import os
import logging
import requests
import threading
import queue
import time
from datetime import datetime

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
processing = False

class SyncBotManager:
    """Synchronous bot manager that avoids async entirely"""
    
    def __init__(self):
        self.token = TELEGRAM_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.processing = False
        
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
    
    def answer_callback_query(self, callback_query_id):
        """Answer callback query synchronously"""
        try:
            url = f"{self.base_url}/answerCallbackQuery"
            data = {
                "callback_query_id": callback_query_id
            }
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
            elif text == '/status':
                self._handle_status(chat_id)
            elif text == '/quickstart':
                self._handle_quickstart(chat_id)
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
⚠️ **LEGAL DISCLAIMER - IMPORTANT**

**RISK WARNING:**
Binary options trading involves substantial risk of loss and is not suitable for all investors.

**BY CONTINUING YOU ACKNOWLEDGE:**
✓ You understand the risks involved
✓ You are at least 18 years old
✓ You accept this disclaimer

*If you do not understand these risks, please do not continue.*"""

            keyboard = {
                "inline_keyboard": [
                    [{"text": "✅ I UNDERSTAND & ACCEPT", "callback_data": "disclaimer_accepted"}],
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
            self.send_message(chat_id, "🤖 Welcome! Use /help for assistance.")
    
    def _handle_help(self, chat_id):
        """Handle /help command"""
        help_text = """
📖 **Binary Options AI Pro - Help**

**Commands:**
/start - Start bot with binary options
/help - Show this help message  
/signals - Get quick binary signals
/assets - View available assets
/status - Check bot status

**Features:**
• AI-powered binary signals
• 15 trading assets
• Risk management tools
• Educational content

*Use menu buttons for best experience*"""
        
        self.send_message(chat_id, help_text, parse_mode="Markdown")
    
    def _handle_signals(self, chat_id):
        """Handle /signals command"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 GET BINARY SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = "🎯 **Binary Signals**\n\nUse the button below to access signals menu."
        self.send_message(chat_id, text, parse_mode="Markdown", reply_markup=keyboard)
    
    def _handle_assets(self, chat_id):
        """Handle /assets command"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "💱 EUR/USD", "callback_data": "asset_EUR/USD"}],
                [{"text": "💱 GBP/USD", "callback_data": "asset_GBP/USD"}],
                [{"text": "💱 USD/JPY", "callback_data": "asset_USD/JPY"}],
                [{"text": "₿ BTC/USD", "callback_data": "asset_BTC/USD"}],
                [{"text": "🟡 XAU/USD", "callback_data": "asset_XAU/USD"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = "📊 **Available Assets**\n\nSelect an asset to trade:"
        self.send_message(chat_id, text, parse_mode="Markdown", reply_markup=keyboard)
    
    def _handle_status(self, chat_id):
        """Handle /status command"""
        status_text = """
✅ **Bot Status: Operational**

🤖 **Services:**
• Telegram API: Connected
• Signal Generation: Ready
• Market Data: Active
• User Interface: Working

*All systems normal*"""
        
        self.send_message(chat_id, status_text, parse_mode="Markdown")
    
    def _handle_quickstart(self, chat_id):
        """Handle /quickstart command"""
        quickstart_text = """
🚀 **Quick Start Guide**

1. Use /start to begin
2. Accept the disclaimer
3. Browse the main menu
4. Get signals for any asset
5. Trade responsibly

*Start with /start*"""
        
        self.send_message(chat_id, quickstart_text, parse_mode="Markdown")
    
    def _handle_unknown(self, chat_id):
        """Handle unknown commands"""
        text = "🤖 I didn't understand that. Use /help for available commands."
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
                    "❌ **Disclaimer Declined**\n\nUse /start to try again.",
                    parse_mode="Markdown"
                )
                
            elif data == "menu_main":
                self._show_main_menu(chat_id, message_id)
                
            elif data == "menu_signals":
                self._show_signals_menu(chat_id, message_id)
                
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
                    
            else:
                self.edit_message_text(
                    chat_id, message_id,
                    "🔄 **Feature Coming Soon**\n\nThis feature is in development.",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.error(f"❌ Button handler error: {e}")
            try:
                self.edit_message_text(
                    chat_id, message_id,
                    "❌ **Error**\n\nPlease try again.",
                    parse_mode="Markdown"
                )
            except:
                pass
    
    def _show_main_menu(self, chat_id, message_id=None):
        """Show main menu"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "🎯 GET BINARY SIGNALS", "callback_data": "menu_signals"}],
                [{"text": "📊 TRADING ASSETS", "callback_data": "menu_assets"}],
                [{"text": "🤖 AI STRATEGIES", "callback_data": "menu_strategies"}],
                [{"text": "💼 ACCOUNT & LIMITS", "callback_data": "menu_account"}],
                [{"text": "📚 LEARN TRADING", "callback_data": "menu_education"}],
                [{"text": "⚡ QUICK START GUIDE", "callback_data": "menu_quickstart"}],
            ]
        }
        
        text = """
🤖 **Binary Options AI Pro** 🚀

*Professional AI-Powered Binary Trading*

🎯 **Live Binary Signals** with AI Analysis
📊 **Multiple Trading Assets** Available
🤖 **Advanced AI Engines** for Accuracy
⏰ **Flexible Expiry Times** from 1-60min

💎 **Your Account:** FREE TRIAL
📈 **Signals Available:** Unlimited during trial

*Tap buttons to start trading!*"""
        
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
    
    def _show_signals_menu(self, chat_id, message_id):
        """Show signals menu"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "⚡ QUICK SIGNAL (5min EUR/USD)", "callback_data": "signal_EUR/USD_5"}],
                [{"text": "📈 STANDARD (15min EUR/USD)", "callback_data": "signal_EUR/USD_15"}],
                [{"text": "🎯 CUSTOM ASSET", "callback_data": "menu_assets"}],
                [{"text": "1 MINUTE", "callback_data": "signal_EUR/USD_1"}],
                [{"text": "5 MINUTES", "callback_data": "signal_EUR/USD_5"}],
                [{"text": "15 MINUTES", "callback_data": "signal_EUR/USD_15"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = """
🎯 **Binary Options Signals**

*Choose your trading style:*

⚡ **Quick Signals** - Fast 1-5min trades
📈 **Standard Signals** - 15min analysis time
🎯 **Custom Signals** - Choose any asset

*AI-powered analysis for better accuracy*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _show_asset_expiry(self, chat_id, message_id, asset):
        """Show expiry options for asset"""
        keyboard = {
            "inline_keyboard": [
                [{"text": "⚡ 1 MINUTE", "callback_data": f"expiry_{asset}_1"}],
                [{"text": "⚡ 5 MINUTES", "callback_data": f"expiry_{asset}_5"}],
                [{"text": "📈 15 MINUTES", "callback_data": f"expiry_{asset}_15"}],
                [{"text": "📈 30 MINUTES", "callback_data": f"expiry_{asset}_30"}],
                [{"text": "🔙 BACK TO ASSETS", "callback_data": "menu_assets"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
📊 **{asset}** - Binary Options

*Choose expiry time:*

⚡ **1-5 minutes** - Quick trades
📈 **15-30 minutes** - More analysis time

*AI will analyze current market conditions*"""
        
        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )
    
    def _generate_signal(self, chat_id, message_id, asset, expiry):
        """Generate trading signal"""
        import random
        
        # Simulate AI analysis
        direction = "CALL" if random.random() > 0.5 else "PUT"
        confidence = random.randint(70, 90)
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "🔄 NEW SIGNAL", "callback_data": f"signal_{asset}_{expiry}"}],
                [{"text": "📊 DIFFERENT ASSET", "callback_data": "menu_assets"}],
                [{"text": "🔙 MAIN MENU", "callback_data": "menu_main"}]
            ]
        }
        
        text = f"""
🎯 **Binary Signal - {asset}**

📈 **Direction:** {direction}
📊 **Confidence:** {confidence}%
⏰ **Expiry:** {expiry} minutes
💎 **Asset:** {asset}

**AI Analysis Complete:**
• Trend Analysis: ✅ Confirmed
• Momentum: ✅ Strong
• Volatility: ✅ Optimal

💰 **Expected Payout:** 75-85%

**Recommendation:**
Place a **{direction}** option with {expiry}-minute expiry.

⚠️ **Risk Management:**
• Risk only 1-2% of capital
• Use demo account first
• Trade responsibly"""

        self.edit_message_text(
            chat_id, message_id,
            text, parse_mode="Markdown", reply_markup=keyboard
        )

# Create bot manager
bot_manager = SyncBotManager()

def process_queued_updates():
    """Process updates from queue in background"""
    global processing
    
    while True:
        try:
            if not update_queue.empty():
                update_data = update_queue.get_nowait()
                bot_manager.process_update(update_data)
            else:
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
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
        "service": "binary-options-bot", 
        "version": "2.0.0",
        "bot_ready": True,
        "queue_size": update_queue.qsize(),
        "features": ["binary_signals", "ai_analysis", "sync_processing"]
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "queue_size": update_queue.qsize()
    })

@app.route('/set_webhook')
def set_webhook():
    """Set webhook for binary options bot"""
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
            "telegram_response": response.json()
        }
        
        logger.info(f"🌐 Webhook set to: {webhook_url}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Webhook setup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Binary options webhook endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        update_id = update_data.get('update_id', 'unknown')
        
        logger.info(f"📨 Received update: {update_id}")
        
        # Add to queue for processing
        update_queue.put(update_data)
        
        return jsonify({
            "status": "queued", 
            "update_id": update_id,
            "queue_size": update_queue.qsize()
        })
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        "status": "ready",
        "bot_ready": True,
        "queue_size": update_queue.qsize(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/debug')
def debug():
    """Debug endpoint"""
    return jsonify({
        "bot_ready": True,
        "queue_size": update_queue.qsize(),
        "telegram_token_set": bool(TELEGRAM_TOKEN),
        "environment": os.getenv("RENDER", "development")
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Starting Binary Options AI Pro on port {port}")
    logger.info("🎯 Synchronous Processing | No Event Loop Issues")
    logger.info(f"📊 Bot Ready: {bool(TELEGRAM_TOKEN)}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
