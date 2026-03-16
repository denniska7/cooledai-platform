# CooledAI Backend - Railway deployment
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (minimal - no torch/streamlit)
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy project (modular structure: api, services, reports, core, backend)
COPY api/ ./api/
COPY services/ ./services/
COPY reports/ ./reports/
COPY core/ ./core/
COPY backend/ ./backend/

# Run FastAPI (from project root so api.main is found)
WORKDIR /app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
