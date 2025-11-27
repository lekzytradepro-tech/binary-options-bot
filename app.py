from flask import Flask, request, jsonify
import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Telegram application
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if TELEGRAM_TOKEN:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
else:
    application = None
    logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set")

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
        
        # Process using proper Telegram Update object
        if application:
            update = Update.de_json(update_data, application.bot)
            asyncio.run(process_update(update))
        else:
            # Fallback processing
            asyncio.run(process_update_fallback(update_data))
        
        return jsonify({"status": "processed"})
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

async def process_update(update: Update):
    """Process update using proper Telegram objects"""
    try:
        context = ContextTypes.DEFAULT_TYPE(application)
        
        if update.message and update.message.text:
            text = update.message.text
            
            if text == '/start':
                from src.bot.handlers import handle_start
                await handle_start(update, context)
            elif text == '/help':
                from src.bot.handlers import handle_help  
                await handle_help(update, context)
            elif text == '/signals':
                from src.bot.handlers import handle_signals
                await handle_signals(update, context)
            elif text == '/assets':
                from src.bot.handlers import handle_assets
                await handle_assets(update, context)
            else:
                from src.bot.handlers import handle_unknown
                await handle_unknown(update, context)
                
        elif update.callback_query:
            from src.bot.handlers import handle_button_click
            await handle_button_click(update, context)
            
    except Exception as e:
        logger.error(f"❌ Update processing error: {e}")

async def process_update_fallback(update_data: dict):
    """Fallback processing for when Telegram app isn't available"""
    try:
        if 'message' in update_data:
            message = update_data['message']
            text = message.get('text', '')
            chat_id = message['chat']['id']
            
            # Create a minimal update-like object
            class SimpleUpdate:
                def __init__(self, data):
                    self.effective_user = type('User', (), {
                        'id': data['chat']['id'],
                        'username': data['chat'].get('username'),
                        'first_name': data['chat'].get('first_name', 'User')
                    })()
                    self.message = type('Message', (), {
                        'text': data.get('text', ''),
                        'chat': type('Chat', (), {'id': data['chat']['id']})(),
                        'reply_text': self.reply_text
                    })()
                    
                async def reply_text(self, text, parse_mode=None, reply_markup=None):
                    # Implement basic reply functionality
                    import requests
                    token = os.getenv("TELEGRAM_BOT_TOKEN")
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    data = {
                        "chat_id": self.effective_user.id,
                        "text": text,
                        "parse_mode": parse_mode
                    }
                    if reply_markup:
                        data["reply_markup"] = reply_markup.to_dict()
                    requests.post(url, json=data)
            
            if 'callback_query' in update_data:
                # Handle callback query
                callback_data = update_data['callback_query']
                class CallbackUpdate:
                    def __init__(self, data):
                        self.callback_query = type('CallbackQuery', (), {
                            'data': data.get('data', ''),
                            'edit_message_text': self.edit_message_text,
                            'answer': self.answer
                        })()
                        self.effective_user = type('User', (), {
                            'id': data['from']['id'],
                            'username': data['from'].get('username'),
                            'first_name': data['from'].get('first_name', 'User')
                        })()
                        
                    async def edit_message_text(self, text, parse_mode=None, reply_markup=None):
                        import requests
                        token = os.getenv("TELEGRAM_BOT_TOKEN")
                        url = f"https://api.telegram.org/bot{token}/editMessageText"
                        data = {
                            "chat_id": update_data['callback_query']['message']['chat']['id'],
                            "message_id": update_data['callback_query']['message']['message_id'],
                            "text": text,
                            "parse_mode": parse_mode
                        }
                        if reply_markup:
                            data["reply_markup"] = reply_markup.to_dict()
                        requests.post(url, json=data)
                        
                    async def answer(self):
                        import requests
                        token = os.getenv("TELEGRAM_BOT_TOKEN")
                        url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
                        data = {
                            "callback_query_id": update_data['callback_query']['id']
                        }
                        requests.post(url, json=data)
                
                update = CallbackUpdate(update_data['callback_query'])
                from src.bot.handlers import handle_button_click
                await handle_button_click(update, None)
                
            else:
                # Handle regular message
                update = SimpleUpdate(message)
                
                if text == '/start':
                    from src.bot.handlers import handle_start
                    await handle_start(update, None)
                elif text == '/help':
                    from src.bot.handlers import handle_help  
                    await handle_help(update, None)
                elif text == '/signals':
                    from src.bot.handlers import handle_signals
                    await handle_signals(update, None)
                elif text == '/assets':
                    from src.bot.handlers import handle_assets
                    await handle_assets(update, None)
                else:
                    from src.bot.handlers import handle_unknown
                    await handle_unknown(update, None)
                    
    except Exception as e:
        logger.error(f"❌ Fallback processing error: {e}")

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
