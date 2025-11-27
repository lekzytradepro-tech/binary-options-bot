FROM python:3.11-slim

WORKDIR /web

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

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
