FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose service port
EXPOSE 8032

# Health check — uses $PORT with fallback so it works both locally and on Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8032') + '/health')" || exit 1

# Run application — shell form so $PORT is expanded from Railway env
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8032}
