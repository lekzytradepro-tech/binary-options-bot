#!/usr/bin/env python3
"""
Health server for Render port binding
Render requires an open port, so we run a simple health check server
"""

from flask import Flask
import threading
import logging

logger = logging.getLogger(__name__)

def create_health_app():
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
    
    return app

def start_health_server(port=8000):
    """Start health server in a separate thread"""
    app = create_health_app()
    
    def run_server():
        logger.info(f"Starting health server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    
    # Start in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("Health server started")
    
    return server_thread
