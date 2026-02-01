"""
CooledAI Backend Entry Point

Re-exports the FastAPI app for deployment (Procfile: uvicorn main:app).
"""

from api.main import app

__all__ = ["app"]
