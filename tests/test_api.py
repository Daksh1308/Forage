import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")

from forge_ai import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "Forge" in data["system"]


def test_create_task(client):
    response = client.post("/research/task", json={
        "task_name": "test-task-1",
        "description": "Write hello world in Python"
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "created"


def test_create_task_missing_name(client):
    response = client.post("/research/task", json={
        "description": "No name provided"
    })
    assert response.status_code == 422


def test_get_history_empty(client):
    response = client.get("/research/history")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data


def test_get_discoveries(client):
    response = client.get("/research/discoveries")
    assert response.status_code == 200
    data = response.json()
    assert "discoveries" in data


def test_get_pending_tasks(client):
    response = client.get("/tasks/pending")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data


def test_register_worker(client):
    response = client.post("/workers/register", json={
        "worker_id": "test-worker-1"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "registered"


def test_worker_stats(client):
    response = client.get("/workers/stats")
    assert response.status_code == 200
    data = response.json()
    assert "online_workers" in data
    assert "completed_tasks" in data


def test_worker_heartbeat(client):
    client.post("/workers/register", json={"worker_id": "hb-worker"})
    response = client.post("/workers/heartbeat", json={"worker_id": "hb-worker"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["worker_id"] == "hb-worker"
