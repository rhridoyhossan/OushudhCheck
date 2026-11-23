FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required by OpenCV and RapidOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for FastAPI Backend & Telegram Bot
COPY backend/requirements.txt /app/backend/requirements.txt
COPY bot/requirements.txt /app/bot/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/backend/requirements.txt -r /app/bot/requirements.txt

# Copy application source and processed SQLite database
COPY backend /app/backend
COPY bot /app/bot
COPY data/processed /app/data/processed

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=25s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
