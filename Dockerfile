FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (REMOVE SUPERVISOR)
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

# Create necessary directories
RUN mkdir -p /app/data

# Create non-root user
RUN useradd -m -r botuser && \
    chown -R botuser:botuser /app
USER botuser

EXPOSE 8000

# Use simple python command (REMOVE SUPERVISOR)
CMD ["python", "render_start.py"]
