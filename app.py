#!/usr/bin/env python3
"""
Binary Options AI Pro - Webhook Server
PRODUCTION READY FOR RENDER
"""

import os
import asyncio
from flask import Flask, request, jsonify
from src.bot.webhook_bot import WebhookBot

# Create Flask app
app = Flask(__name__)

# Initialize bot
bot = WebhookBot()

@app.before_first_request
def initialize_bot():
    """Initialize bot on first request"""
    try:
        # Run async initialization
        asyncio.run(bot.initialize())
        print("✅ Bot initialized successfully")
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "binary-options-bot", 
        "version": "2.0.0",
        "webhook": "active"
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "bot": "initialized"})

@app.route("/webhook", methods=["POST"])
def webhook():
    """Telegram webhook endpoint"""
    try:
        if request.is_json:
            update_data = request.get_json()
            
            # Process update in async context
            async def process():
                await bot.process_update(update_data)
            
            asyncio.run(process())
            return jsonify({"status": "processed"})
        else:
            return jsonify({"error": "Invalid content type"}), 400
            
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/set_webhook")
def set_webhook():
    """Manual webhook setup endpoint"""
    try:
        asyncio.run(bot.set_webhook())
        return jsonify({
            "status": "success", 
            "message": "Webhook set successfully",
            "url": bot.webhook_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/info")
def bot_info():
    """Get bot information"""
    try:
        return jsonify({
            "bot_username": "BinaryOptionsAIPro",
            "status": "active",
            "webhook_url": bot.webhook_url,
            "features": ["ai_signals", "trading_strategies", "education"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Starting Binary Options AI Pro on port {port}")
    print(f"🔗 Webhook URL: {os.getenv('WEBHOOK_URL', 'Not set')}")
    
    # Initialize bot before starting server
    initialize_bot()
    
    app.run(host="0.0.0.0", port=port, debug=False)
