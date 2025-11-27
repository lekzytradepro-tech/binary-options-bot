FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including supervisor
RUN apt-get update && apt-get install -y \
    gcc \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Create non-root user
RUN useradd -m -r botuser && \
    chown -R botuser:botuser /app
USER botuser

EXPOSE 8000

# Use supervisor to manage processes instead of direct python command
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
