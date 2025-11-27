FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -r botuser && \
    chown -R botuser:botuser /app
USER botuser

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "main.py"]
