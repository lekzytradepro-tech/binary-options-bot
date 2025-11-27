#!/usr/bin/env python3
"""
Binary Options AI Pro - FIXED Async Version
"""

import os
import asyncio
import logging
from flask import Flask, request, jsonify
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global event loop for async operations
event_loop = None

def get_event_loop():
    """Get or create event loop"""
    global event_loop
    if event_loop is None:
        try:
            event_loop = asyncio.get_event_loop()
        except RuntimeError:
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
    return event_loop

@app.route("/")
def home():
    return jsonify({
        "status": "running", 
        "service": "binary-options-bot",
        "version": "2.0.0"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/set_webhook")
def set_webhook():
    """Set webhook manually"""
    import requests
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    webhook_url = "https://binary-options-bot-4c74.onrender.com/webhook"
    
    if not token:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
    
    # Set webhook
    url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
    response = requests.get(url)
    
    return jsonify({
        "status": "webhook_set",
        "webhook_url": webhook_url,
        "telegram_response": response.json()
    })

@app.route("/webhook", methods=["POST"])
def webhook():
    """Webhook endpoint with proper async handling"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        logger.info(f"📨 Received update: {update_data.get('update_id', 'unknown')}")
        
        # Process in thread to avoid async issues
        def process_sync():
            loop = get_event_loop()
            return asyncio.run_coroutine_threadsafe(
                process_telegram_update(update_data), 
                loop
            ).result(timeout=10)  # 10 second timeout
            
        result = process_sync()
        return jsonify({"status": "processed", "result": result})
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

async def process_telegram_update(update_data):
    """Process Telegram update with proper bot initialization"""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
        
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        
        # Create application instance
        application = Application.builder().token(token).build()
        
        # Add command handlers
        async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user = update.effective_user
            await update.message.reply_text(
                f"🤖 **Binary Options AI Pro**\n\nHello {user.first_name}! 🎉\n\nBot is working perfectly! 🚀",
                parse_mode="Markdown"
            )
            logger.info(f"✅ Responded to user {user.id}")
        
        async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "📖 **Help**\n\n/start - Start the bot\n/help - This message\n\nMore features coming soon!",
                parse_mode="Markdown"
            )
        
        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        
        # Initialize and process
        await application.initialize()
        
        update = Update.de_json(update_data, application.bot)
        await application.process_update(update)
        
        # Cleanup
        await application.shutdown()
        
        return "success"
        
    except Exception as e:
        logger.error(f"❌ Process update error: {e}")
        return f"error: {e}"

@app.route("/test")
def test_bot():
    """Test if bot can send messages"""
    import requests
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
    
    # Get bot info to test token
    url = f"https://api.telegram.org/bot{token}/getMe"
    response = requests.get(url)
    
    if response.status_code == 200:
        bot_info = response.json()
        return jsonify({
            "status": "bot_connected",
            "bot_username": bot_info['result']['username'],
            "bot_name": bot_info['result']['first_name']
        })
    else:
        return jsonify({
            "status": "bot_error", 
            "error": response.json()
        }), 500

# Initialize event loop on startup
get_event_loop()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    print(f"🔗 Webhook URL: https://binary-options-bot-4c74.onrender.com/webhook")
    
    app.run(host="0.0.0.0", port=port, debug=False)
