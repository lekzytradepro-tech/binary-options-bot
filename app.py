from flask import Flask, request, jsonify
import os
import logging
import asyncio
from src.bot.handlers import handle_button_click

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
    webhook_url = "https://binary-options-bot-4c74.onrender.com/webhook"
    
    if not token:
        return jsonify({"error": "TELEGRAM_BOT_TOKEN not set"}), 500
    
    url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
    response = requests.get(url)
    
    return jsonify({
        "status": "webhook_set",
        "webhook_url": webhook_url,
        "telegram_response": response.json()
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Binary options webhook endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400
            
        update_data = request.get_json()
        logger.info(f"📨 Binary update: {update_data.get('update_id', 'unknown')}")
        
        # Process using new binary bot handlers
        if 'message' in update_data:
            message = update_data['message']
            text = message.get('text', '')
            chat_id = message['chat']['id']
            
            if text == '/start':
                from src.bot.handlers import handle_start
                asyncio.run(handle_start(message, None))
            elif text == '/help':
                from src.bot.handlers import handle_help  
                asyncio.run(handle_help(message, None))
            elif text == '/signals':
                from src.bot.handlers import handle_signals
                asyncio.run(handle_signals(message, None))
            elif text == '/assets':
                from src.bot.handlers import handle_assets
                asyncio.run(handle_assets(message, None))
            else:
                from src.bot.handlers import handle_unknown
                asyncio.run(handle_unknown(message, None))
                
        elif 'callback_query' in update_data:
            asyncio.run(handle_button_click(update_data['callback_query'], None))
        
        return jsonify({"status": "processed"})
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test')
def test():
    """Test binary options features"""
    try:
        from src.core.config import Config
        from src.api.twelvedata_client import TwelveDataClient
        
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
