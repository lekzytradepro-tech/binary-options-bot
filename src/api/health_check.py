#!/usr/bin/env python3
"""
Health check service for supervisor monitoring
"""

import time
import requests
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            logger.info("Health check passed")
        else:
            self.send_response(404)
            self.end_headers()

def run_health_check():
    """Run simple health check server"""
    server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    logger.info("Health check server starting on port 8080...")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Health check server stopping...")
    finally:
        server.server_close()

if __name__ == '__main__':
    run_health_check()
