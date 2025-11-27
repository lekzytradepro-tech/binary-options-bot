from flask import Flask, request, jsonify
import os
import logging
import asyncio
import threading
import queue
from telegram import Update
from telegram.ext import Application, ContextTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
application = None
bot_initialized = False
first_request = True
update_queue = queue.Queue()

class BotManager:
    """Manage Telegram bot with proper event loop handling"""
    
    def __init__(self):
        self.application = None
        self.initialized = False
        self.loop = None
        self.thread = None
        
    def initialize_bot(self):
        """Initialize the bot in a dedicated thread"""
        try:
            token = os.getenv("TELEGRAM_BOT_TOKEN")
            if not token:
                logger.error("❌ TELEGRAM_BOT_TOKEN not found")
                return False
            
            # Start bot in a dedicated thread
            self.thread = threading.Thread(target=self._run_bot, daemon=True)
            self.thread.start()
            
            # Wait for initialization
            for i in range(30):  # 30 second timeout
                if self.initialized:
                    logger.info("✅ Bot initialized successfully")
                    return True
                threading.Event().wait(1)  # Wait 1 second
                
            logger.error("❌ Bot initialization timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            return False
    
    def _run_bot(self):
        """Run the bot in a dedicated thread with its own event loop"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the bot setup
            self.loop.run_until_complete(self._setup_bot())
            
            # Start processing updates
            self.loop.run_until_complete(self._process_updates())
            
        except Exception as e:
            logger.error(f"❌ Bot thread error: {e}")
    
    async def _setup_bot(self):
        """Set up the bot application"""
        try:
            token = os.getenv("TELEGRAM_BOT_TOKEN")
            
            # Create application
            self.application = (
                Application.builder()
                .token(token)
                .build()
            )
            
            # Add handlers
            from src.bot.handlers import (
                handle_start, handle_help, handle_signals, 
                handle_assets, handle_button_click, handle_unknown,
                handle_status, handle_quick_start
            )
            from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, filters
            
            # Clear any existing handlers
            self.application.handlers.clear()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", handle_start))
            self.application.add_handler(CommandHandler("help", handle_help))
            self.application.add_handler(CommandHandler("signals", handle_signals))
            self.application.add_handler(CommandHandler("assets", handle_assets))
            self.application.add_handler(CommandHandler("status", handle_status))
            self.application.add_handler(CommandHandler("quickstart", handle_quick_start))
            self.application.add_handler(CallbackQueryHandler(handle_button_click))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))
            
            # Initialize
            await self.application.initialize()
            
            self.initialized = True
            logger.info(f"📊 Bot setup complete - {len(self.application.handlers[0])} handlers registered")
            
        except Exception as e:
            logger.error(f"❌ Bot setup failed: {e}")
            self.initialized = False
    
    async def _process_updates(self):
        """Process updates from the queue"""
        while True:
            try:
                # Check for updates in queue
                if not update_queue.empty():
                    update_data = update_queue.get_nowait()
                    await self._handle_update(update_data)
                else:
                    # Small sleep to prevent busy waiting
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Update processing error: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _handle_update(self, update_data):
        """Handle a single update"""
        try:
            if not self.application:
                logger.error("❌ Application not available")
                return
                
            # Create Update object
            update = Update.de_json(update_data, self.application.bot)
            
            # Process the update
            await self.application.process_update(update)
            
            logger.info(f"✅ Processed update: {update.update_id}")
            
        except Exception as e:
            logger.error(f"❌ Error handling update: {e}")

# Create bot manager instance
bot_manager = BotManager()

@app.before_request
def initialize_on_first_request():
    """Initialize bot on first request"""
    global first_request
    
    if first_request:
        first_request = False
        logger.info("🚀 First request received - initializing bot...")
        
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            if bot_manager.initialize_bot():
                logger.info("✅ Bot manager initialized successfully")
            else:
                logger.error("❌ Bot manager initialization failed")
        else:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set")

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "binary-options-bot", 
        "version": "2.0.0",
        "bot_initialized": bot_manager.initialized,
        "features": ["binary_signals", "ai_analysis", "real_data", "multiple_assets"]
    })

@app.route('/health')
def health():
    status = "healthy" if bot_manager.initialized else "degraded"
    return jsonify({
        "status": status,
        "bot_initialized": bot_manager.initialized,
        "timestamp": os.getenv("RENDER_GIT_COMMIT_TIMESTAMP", "unknown")
    })

@app.route('/set_webhook')
def set_webhook():
    """Set webhook for binary options bot"""
    try:
        import requests
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        webhook_url = os.getenv("WEBHOOK_URL", "https://your-app-name.onrender.com/webhook")
        
        if not token:
            return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
        
        url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
        response = requests.get(url, timeout=10)
        
        result = {
            "status": "webhook_set",
            "webhook_url": webhook_url,
            "bot_initialized": bot_manager.initialized,
            "telegram_response": response.json()
        }
        
        logger.info(f"🌐 Webhook set to: {webhook_url}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Webhook setup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Binary options webhook endpoint - simple and reliable"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        update_id = update_data.get('update_id', 'unknown')
        
        logger.info(f"📨 Received update: {update_id}")
        
        # Check if bot is initialized
        if not bot_manager.initialized:
            logger.error("❌ Bot not initialized")
            return jsonify({"error": "Bot not initialized"}), 503
        
        # Add update to queue for processing
        update_queue.put(update_data)
        
        logger.info(f"✅ Update {update_id} queued for processing")
        
        return jsonify({
            "status": "queued", 
            "update_id": update_id,
            "bot_initialized": bot_manager.initialized
        })
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/initialize', methods=['POST'])
def initialize_bot_endpoint():
    """Endpoint to manually initialize the bot"""
    try:
        success = bot_manager.initialize_bot()
        status = "initialized" if success else "failed"
        return jsonify({
            "status": status,
            "bot_initialized": bot_manager.initialized
        })
    except Exception as e:
        logger.error(f"❌ Manual initialization failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    """Test endpoint"""
    try:
        from src.core.config import Config
        
        return jsonify({
            "status": "ready",
            "bot_initialized": bot_manager.initialized,
            "ai_engines": Config.BINARY_AI_ENGINES,
            "trading_assets": Config.BINARY_PAIRS,
            "binary_expiries": Config.BINARY_EXPIRIES,
            "queue_size": update_queue.qsize()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug')
def debug():
    """Debug endpoint"""
    return jsonify({
        "bot_initialized": bot_manager.initialized,
        "application_exists": bot_manager.application is not None,
        "queue_size": update_queue.qsize(),
        "telegram_token_set": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "environment": os.getenv("RENDER", "development")
    })

# Simple direct update processing as fallback
def process_update_directly(update_data):
    """Process update directly without complex async handling"""
    try:
        import requests
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        if not token:
            logger.error("❌ No bot token available")
            return
            
        # Extract basic info from update
        if 'message' in update_data:
            message = update_data['message']
            chat_id = message['chat']['id']
            text = message.get('text', '')
            
            # Handle basic commands directly
            if text == '/start':
                response_text = """
🤖 **Binary Options AI Pro** 🚀

Welcome! I'm your AI-powered binary options assistant.

Please use the menu buttons to navigate, or try these commands:

/help - Get help information
/signals - Get trading signals  
/assets - View available assets
/status - Check bot status

*Use buttons for the best experience!*"""
                
            elif text == '/help':
                response_text = """
📖 **Binary Options Help**

Available Commands:
/start - Start the bot
/help - This help message
/signals - Get trading signals
/assets - View trading assets
/status - Check system status

*For full features, please use the menu buttons.*"""
                
            elif text == '/signals':
                response_text = "🎯 **Signals Feature**\n\nPlease use the menu buttons to access signals with full AI analysis."
                
            elif text == '/assets':
                response_text = "📊 **Trading Assets**\n\nPlease use the menu buttons to view all available assets."
                
            else:
                response_text = "🤖 I didn't understand that command. Use /help for available commands."
            
            # Send response directly via Telegram API
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": response_text,
                "parse_mode": "Markdown"
            }
            requests.post(url, json=data, timeout=10)
            
        elif 'callback_query' in update_data:
            # Handle callback queries directly
            callback = update_data['callback_query']
            callback_id = callback['id']
            chat_id = callback['message']['chat']['id']
            data = callback.get('data', '')
            
            # Answer callback first
            answer_url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
            requests.post(answer_url, json={"callback_query_id": callback_id}, timeout=5)
            
            # Send simple response
            response_url = f"https://api.telegram.org/bot{token}/sendMessage"
            response_data = {
                "chat_id": chat_id,
                "text": "🔄 **Feature Loading**\n\nPlease use /start to access the full menu interface.",
                "parse_mode": "Markdown"
            }
            requests.post(response_url, json=response_data, timeout=10)
            
    except Exception as e:
        logger.error(f"❌ Direct processing error: {e}")

@app.route('/simple_webhook', methods=['POST'])
def simple_webhook():
    """Simple webhook that processes updates directly"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        
        # Process directly in this thread
        process_update_directly(update_data)
        
        return jsonify({"status": "processed"})
        
    except Exception as e:
        logger.error(f"❌ Simple webhook error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    logger.info(f"🚀 Starting Binary Options AI Pro on port {port}")
    logger.info("🎯 8 AI Engines | 15 Assets | Real TwelveData")
    
    # Initialize bot if token is available
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.info("🤖 Initializing bot manager...")
        bot_manager.initialize_bot()
    
    app.run(host='0.0.0.0', port=port, debug=False)
