"""
main.py
────────
FastAPI wrapper for the fraud inference script.
Deployed on Railway so the Vercel app can trigger scoring from anywhere.

Endpoints:
  GET  /          — health check
  POST /score     — runs run_inference.py and returns stdout/stderr

Deploy:
  railway up  (from this directory)

Environment variables to set in Railway dashboard:
  SUPABASE_URL = https://tqvaebgxkymimisiahfc.supabase.co
  SUPABASE_KEY = sb_secret_...
"""

import subprocess
import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Fraud Inference Service")

# Allow the Vercel frontend to call this from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ScoringResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    duration_ms: int


@app.get("/")
def health():
    """Health check — Railway uses this to confirm the service is up."""
    return {"status": "ok", "service": "fraud-inference"}


@app.post("/score", response_model=ScoringResult)
def score():
    """
    Trigger the fraud inference job.
    Runs run_inference.py as a subprocess and returns its output.
    """
    import time
    start = time.time()

    script_path = os.path.join(os.path.dirname(__file__), "run_inference.py")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute max
            env={
                **os.environ,
                "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
                "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
            },
        )
        duration_ms = int((time.time() - start) * 1000)
        return ScoringResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start) * 1000)
        return ScoringResult(
            success=False,
            stdout="",
            stderr="Inference timed out after 5 minutes.",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        return ScoringResult(
            success=False,
            stdout="",
            stderr=str(e),
            duration_ms=duration_ms,
        )
