# CooledAI Backend - Railway deployment
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (minimal - no torch/streamlit)
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy project (backend needs core/ from parent)
COPY backend/ ./backend/
COPY core/ ./core/

# Run FastAPI
WORKDIR /app/backend
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
