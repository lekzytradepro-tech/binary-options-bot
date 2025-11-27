from flask import Flask, request, jsonify
import os
import logging
import asyncio
import threading
from telegram import Update
from telegram.ext import Application, ContextTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global application instance
application = None
bot_initialized = False
first_request = True

async def initialize_bot_async():
    """Initialize the Telegram bot application asynchronously"""
    global application, bot_initialized
    
    try:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
            return False
        
        logger.info("🤖 Initializing Telegram Bot Application...")
        
        # Create application with proper settings
        application = (
            Application.builder()
            .token(token)
            .pool_timeout(30)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .build()
        )
        
        # Add all handlers
        from src.bot.handlers import (
            handle_start, handle_help, handle_signals, 
            handle_assets, handle_button_click, handle_unknown,
            handle_status, handle_quick_start
        )
        from telegram.ext import CommandHandler, CallbackQueryHandler, MessageHandler, filters
        
        # Clear any existing handlers
        application.handlers.clear()
        
        # Add handlers in correct order
        application.add_handler(CommandHandler("start", handle_start))
        application.add_handler(CommandHandler("help", handle_help))
        application.add_handler(CommandHandler("signals", handle_signals))
        application.add_handler(CommandHandler("assets", handle_assets))
        application.add_handler(CommandHandler("status", handle_status))
        application.add_handler(CommandHandler("quickstart", handle_quick_start))
        application.add_handler(CallbackQueryHandler(handle_button_click))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))
        
        # Initialize the application properly with await
        await application.initialize()
        
        bot_initialized = True
        logger.info("✅ Telegram Bot Application initialized successfully")
        logger.info(f"📊 Handlers registered: {len(application.handlers[0])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize bot: {e}")
        bot_initialized = False
        return False

def initialize_bot_sync():
    """Initialize bot synchronously for use in Flask context"""
    try:
        # Create new event loop for synchronous context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(initialize_bot_async())
        loop.close()
        return result
    except Exception as e:
        logger.error(f"❌ Sync initialization failed: {e}")
        return False

@app.before_request
def before_first_request():
    """Initialize bot on first request (Flask 2.3+ compatible)"""
    global first_request, bot_initialized
    
    if first_request:
        first_request = False
        logger.info("🚀 First request - initializing bot...")
        
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            if initialize_bot_sync():
                logger.info("✅ Bot initialized successfully on first request")
            else:
                logger.error("❌ Bot initialization failed on first request")
        else:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set, bot disabled")

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "binary-options-bot", 
        "version": "2.0.0",
        "bot_initialized": bot_initialized,
        "features": ["binary_signals", "ai_analysis", "real_data", "multiple_assets"]
    })

@app.route('/health')
def health():
    status = "healthy" if bot_initialized else "degraded"
    return jsonify({
        "status": status,
        "bot_initialized": bot_initialized,
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
            "bot_initialized": bot_initialized,
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
            logger.warning("❌ Received non-JSON webhook request")
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        update_id = update_data.get('update_id', 'unknown')
        logger.info(f"📨 Received update: {update_id}")
        
        # Check if bot is initialized
        if not bot_initialized or not application:
            logger.error("❌ Bot not initialized, cannot process update")
            # Try to reinitialize
            if initialize_bot_sync():
                logger.info("✅ Bot reinitialized successfully")
            else:
                return jsonify({"error": "Bot not initialized"}), 503
        
        # Process update in background thread
        thread = threading.Thread(
            target=process_update_in_thread,
            args=(update_data,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "status": "processing", 
            "update_id": update_id,
            "bot_initialized": bot_initialized
        })
        
    except Exception as e:
        logger.error(f"❌ Webhook processing error: {e}")
        return jsonify({"error": str(e)}), 500

def process_update_in_thread(update_data):
    """Process update in a separate thread with proper event loop management"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Process the update
        loop.run_until_complete(process_update_async(update_data))
        
    except Exception as e:
        logger.error(f"❌ Thread processing error: {e}")
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass

async def process_update_async(update_data):
    """Process update asynchronously"""
    try:
        if not application:
            logger.error("❌ Application not available for processing")
            return
            
        # Create Update object from JSON
        update = Update.de_json(update_data, application.bot)
        
        # Process the update through the application
        await application.process_update(update)
        
        logger.info(f"✅ Successfully processed update: {update.update_id}")
        
    except Exception as e:
        logger.error(f"❌ Async processing error: {e}")

@app.route('/initialize', methods=['POST'])
def initialize_bot_endpoint():
    """Endpoint to manually initialize the bot"""
    try:
        success = initialize_bot_sync()
        status = "initialized" if success else "failed"
        return jsonify({
            "status": status,
            "bot_initialized": bot_initialized,
            "timestamp": os.getenv("RENDER_GIT_COMMIT_TIMESTAMP", "unknown")
        })
    except Exception as e:
        logger.error(f"❌ Manual initialization failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    """Test binary options features"""
    try:
        from src.core.config import Config
        
        return jsonify({
            "status": "binary_bot_ready",
            "bot_initialized": bot_initialized,
            "ai_engines": Config.BINARY_AI_ENGINES,
            "trading_assets": Config.BINARY_PAIRS,
            "binary_expiries": Config.BINARY_EXPIRIES,
            "payout_rates": Config.PAYOUT_RATES,
            "twelvedata_keys": len(Config.TWELVEDATA_KEYS) if hasattr(Config, 'TWELVEDATA_KEYS') else 0
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "bot_initialized": bot_initialized}), 500

@app.route('/debug')
def debug():
    """Debug endpoint to check bot status"""
    try:
        handlers_count = len(application.handlers[0]) if application and application.handlers else 0
        
        return jsonify({
            "bot_initialized": bot_initialized,
            "application_exists": application is not None,
            "handlers_count": handlers_count,
            "telegram_token_set": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
            "environment": os.getenv("RENDER", "development")
        })
    except Exception as e:
        return jsonify({"error": str(e), "bot_initialized": bot_initialized}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    
    # Initialize bot before starting Flask app
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.info("🚀 Starting bot initialization...")
        if initialize_bot_sync():
            logger.info("✅ Bot initialized successfully before Flask start")
        else:
            logger.error("❌ Bot initialization failed before Flask start")
    
    logger.info(f"🚀 Starting Binary Options AI Pro on port {port}")
    logger.info("🎯 8 AI Engines | 15 Assets | Real TwelveData")
    logger.info(f"📊 Bot Initialized: {bot_initialized}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
