from flask import Flask, request, jsonify
import os
import logging
import asyncio
import threading
from telegram import Update
from telegram.ext import Application, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global application instance
application = None

def init_bot():
    """Initialize the Telegram bot application"""
    global application
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if token:
        application = Application.builder().token(token).build()
        
        # Add handlers
        from src.bot.handlers import (
            handle_start, handle_help, handle_signals, 
            handle_assets, handle_button_click, handle_unknown,
            handle_status, handle_quick_start
        )
        from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, filters
        
        application.add_handler(CommandHandler("start", handle_start))
        application.add_handler(CommandHandler("help", handle_help))
        application.add_handler(CommandHandler("signals", handle_signals))
        application.add_handler(CommandHandler("assets", handle_assets))
        application.add_handler(CommandHandler("status", handle_status))
        application.add_handler(CommandHandler("quickstart", handle_quick_start))
        application.add_handler(CallbackQueryHandler(handle_button_click))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))
        
        logger.info("🤖 Telegram bot application initialized")
    else:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set")

# Initialize bot on startup
init_bot()

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "binary-options-bot", 
        "version": "2.0.0",
        "features": ["binary_signals", "ai_analysis", "real_data", "multiple_assets"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/set_webhook')
def set_webhook():
    """Set webhook for binary options bot"""
    import requests
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    webhook_url = os.getenv("WEBHOOK_URL", "https://your-app-name.onrender.com/webhook")
    
    if not token:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
    
    url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
    response = requests.get(url)
    
    return jsonify({
        "status": "webhook_set",
        "webhook_url": webhook_url,
        "telegram_response": response.json()
    })

def run_async(coro):
    """Run async function in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Binary options webhook endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        logger.info(f"📨 Binary update: {update_data.get('update_id', 'unknown')}")
        
        # Process update in a thread with proper event loop
        thread = threading.Thread(target=process_update_thread, args=(update_data,))
        thread.start()
        
        return jsonify({"status": "processing"})
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

def process_update_thread(update_data):
    """Process update in a separate thread with its own event loop"""
    try:
        if application:
            update = Update.de_json(update_data, application.bot)
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the update processing
            loop.run_until_complete(application.process_update(update))
            loop.close()
        else:
            logger.error("❌ Telegram application not initialized")
            
    except Exception as e:
        logger.error(f"❌ Update processing error: {e}")

@app.route('/test')
def test():
    """Test binary options features"""
    try:
        from src.core.config import Config
        
        return jsonify({
            "status": "binary_bot_ready",
            "ai_engines": Config.BINARY_AI_ENGINES,
            "trading_assets": Config.BINARY_PAIRS,
            "binary_expiries": Config.BINARY_EXPIRIES,
            "payout_rates": Config.PAYOUT_RATES,
            "twelvedata_keys": len(Config.TWELVEDATA_KEYS)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"🚀 Starting Binary Options AI Pro on port {port}")
    logger.info("🎯 8 AI Engines | 15 Assets | Real TwelveData")
    app.run(host='0.0.0.0', port=port, debug=False)
