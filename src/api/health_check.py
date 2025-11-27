from flask import Flask, jsonify
import logging

logger = logging.getLogger(__name__)

def create_health_app():
    """Create Flask app for health checks"""
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "service": "binary-options-bot",
            "timestamp": "2024-01-01T00:00:00Z"
        })
    
    @app.route('/')
    def home():
        return jsonify({
            "message": "Binary Options AI Pro Bot",
            "status": "running"
        })
    
    return app

# Start health server in background
def start_health_server(port=8000):
    """Start health server in background thread"""
    import threading
    
    app = create_health_app()
    
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    logger.info(f"Health server started on port {port}")
    
    return thread
