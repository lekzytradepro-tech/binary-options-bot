#!/usr/bin/env python3
"""
Render Web Service - Proper async handling
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

# Create Flask app
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

def run_bot_in_thread():
    """Run bot in separate thread with its own event loop"""
    def start_bot():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            from src.bot.main import main
            loop.run_until_complete(main())
        except Exception as e:
            logger.error(f"Bot thread error: {e}")
    
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    logger.info("Bot started in background thread")
    return bot_thread

def main():
    """Main function - starts bot in background, returns Flask app"""
    logger.info("Starting Binary Options Bot as Web Service...")
    
    # Start bot in background thread
    run_bot_in_thread()
    
    # Return Flask app for Gunicorn
    return app

if __name__ == "__main__":
    # For local development without Gunicorn
    flask_app = main()
    flask_app.run(host='0.0.0.0', port=8000, debug=False)
