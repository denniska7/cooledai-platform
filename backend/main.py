"""
CooledAI Backend Entry Point

Re-exports the FastAPI app for deployment (Procfile: uvicorn main:app).
When run from backend/, project root is added to path so api.main is the root-level api package.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from api.main import app

__all__ = ["app"]
