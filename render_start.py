#!/usr/bin/env python3
"""
Render-specific startup script
Starts both health server and bot
"""

import os
import sys
import threading
import asyncio
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_health_server():
    """Start Flask health server for Render port binding"""
    from flask import Flask
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return '🤖 Binary Options Bot is running!'
    
    @app.route('/health')
    def health():
        return 'OK', 200
    
    @app.route('/status')
    def status():
        return {'status': 'running', 'service': 'binary-options-bot'}, 200
    
    def run_server():
        logger.info("Starting health server on port 8000")
        app.run(host='0.0.0.0', port=8000, debug=False)
    
    # Start in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

async def start_bot():
    """Start the Telegram bot"""
    from src.bot.main import main
    await main()

def main():
    """Main function for Render"""
    logger.info("Starting Binary Options Bot on Render...")
    
    # Start health server first (required for Render port binding)
    start_health_server()
    
    # Start bot
    try:
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise

if __name__ == "__main__":
    main()
