#!/usr/bin/env python3
"""
Simple Flask app for Render web service
Starts bot in separate PROCESS (not thread) to avoid event loop conflicts
"""

from flask import Flask
import multiprocessing
import logging
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def start_bot_process():
    """Start bot in separate PROCESS (not thread)"""
    def run_bot():
        try:
            # Import and run bot in separate process
            from src.bot.main import main
            main()
        except Exception as e:
            logger.error(f"Bot process error: {e}")
    
    # Start bot in separate process
    bot_process = multiprocessing.Process(target=run_bot, daemon=True)
    bot_process.start()
    logger.info(f"Bot started in separate process (PID: {bot_process.pid})")
    return bot_process

# Global reference to keep bot process alive
bot_process = None

@app.before_first_request
def startup():
    """Start bot when first request comes in"""
    global bot_process
    if bot_process is None or not bot_process.is_alive():
        logger.info("Starting bot process...")
        bot_process = start_bot_process()

if __name__ == "__main__":
    # Start bot process when app starts
    startup()
    app.run(host='0.0.0.0', port=8000, debug=False)
