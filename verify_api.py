"""
End-to-end API verification script for PureLearn AI service.
"""

import sys
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def run_checks():
    print("[INFO] Testing GET /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("[SUCCESS] Health check passed:", r.json())

    print("\n[INFO] Testing GET /api/model/info...")
    r = client.get("/api/model/info")
    assert r.status_code == 200, f"Model info failed: {r.text}"
    print("[SUCCESS] Model info passed:", r.json()["model_name"], "| Parameters:", r.json()["total_parameters"])

    print("\n[INFO] Testing POST /api/chat...")
    r = client.post("/api/chat", json={
        "messages": [
            {"role": "user", "content": "How do I take the derivative of f(x) = x^2 * sin(x)?"}
        ],
        "socratic_level": 2,
        "adhd_mode": True,
        "dyslexia_mode": False,
        "bionic_reading": False,
        "temperature": 0.7,
        "max_new_tokens": 100
    })
    assert r.status_code == 200, f"Chat failed: {r.text}"
    print("[SUCCESS] Chat response received:", r.json()["response"][:120], "...")

    print("\n[INFO] Testing POST /api/study-notes...")
    r = client.post("/api/study-notes", json={
        "content": "Newton's Second Law and Momentum in Physics"
    })
    assert r.status_code == 200, f"Study notes failed: {r.text}"
    print("[SUCCESS] Study notes generated successfully.")

    print("\n[INFO] Testing POST /api/flashcards...")
    r = client.post("/api/flashcards", json={
        "content": "Calculus Derivatives and Product Rule",
        "num_cards": 3
    })
    assert r.status_code == 200, f"Flashcards failed: {r.text}"
    print("[SUCCESS] Flashcards generated:", len(r.json()["flashcards"]), "cards.")

    print("\n[PASSED] All PureLearn AI endpoints verified successfully.")

if __name__ == "__main__":
    run_checks()
