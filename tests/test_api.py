from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "CIFAR-10 MLOps Console" in response.text


def test_visualizations():
    response = client.get("/visualizations")
    assert response.status_code == 200
    assert len(response.json()["class_names"]) == 10
