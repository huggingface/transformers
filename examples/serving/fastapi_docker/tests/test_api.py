import os
import subprocess
import sys
import time

import requests


def test_health_and_predict():
    # Start server from the fastapi_docker directory
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8010"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait longer for server to start and model to load
        time.sleep(5)

        # Check health endpoint
        r = requests.get("http://127.0.0.1:8010/health", timeout=10)
        assert r.status_code == 200
        health_data = r.json()
        assert health_data["status"] == "ok"

        # Test prediction endpoint
        payload = {"inputs": ["I love this!", "I hate this."]}
        r2 = requests.post("http://127.0.0.1:8010/predict", json=payload, timeout=30)
        assert r2.status_code == 200
        data = r2.json()
        assert "outputs" in data and len(data["outputs"]) == 2

        print("✅ All tests passed!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
