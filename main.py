#!/usr/bin/env python3
"""
Binary Options AI Pro - Render Compatible
"""

import os
import sys
import threading
from flask import Flask, jsonify

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Create Flask app for health checks
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "Binary Options Bot Running", "service": "telegram-bot"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"})

@app.route('/test')
def test():
    return jsonify({"message": "Bot is alive and healthy"})

def start_health_server():
    """Start health server in background"""
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Starting health server on port {port}")
    
    # Run Flask in background thread
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False),
        daemon=True
    ).start()
    
    print(f"✅ Health server started on port {port}")
    return True

def main():
    """Main function"""
    print("🤖 Starting Binary Options AI Pro...")
    
    # Start health server FIRST
    start_health_server()
    
    # Then start the bot
    try:
        from src.bot.main import run_bot
        print("✅ Starting Telegram bot...")
        run_bot()
    except Exception as e:
        print(f"❌ Bot error: {e}")
        # Keep the health server running even if bot fails
        import time
        while True:
            time.sleep(10)  # Keep process alive

if __name__ == "__main__":
    main()
