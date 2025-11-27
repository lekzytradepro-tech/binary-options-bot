#!/usr/bin/env python3
"""
Render Web Service - Runs both web server and Telegram bot
"""

import os
import sys
import threading
import asyncio
from flask import Flask
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app for Render health checks
app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <html>
        <body>
            <h1>🤖 Binary Options AI Pro</h1>
            <p>Telegram bot is running in background</p>
            <p>Add the bot on Telegram to use it</p>
        </body>
    </html>
    '''

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/status')
def status():
    return {'status': 'running', 'bot': 'active'}, 200

def run_flask():
    """Run Flask app in a separate thread"""
    logger.info("Starting web server on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

async def run_bot():
    """Run the Telegram bot"""
    from src.bot.main import main
    await main()

def main():
    """Main function - starts both web server and bot"""
    logger.info("Starting Binary Options Bot as Web Service...")
    
    # Start Flask web server in background thread
    web_thread = threading.Thread(target=run_flask, daemon=True)
    web_thread.start()
    
    # Start Telegram bot in main thread
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise

if __name__ == "__main__":
    main()
