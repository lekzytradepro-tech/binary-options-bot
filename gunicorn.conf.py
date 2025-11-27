# Gunicorn configuration
bind = "0.0.0.0:10000"
workers = 1
worker_class = "sync"
timeout = 60
keepalive = 5
