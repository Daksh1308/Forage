import pytest
import sys
sys.path.insert(0, ".")

from forge_ai import TaskQueue


@pytest.fixture
def queue():
    q = TaskQueue(":memory:")
    return q


def test_enqueue_task(queue):
    task_id = queue.enqueue_task("test-task", "A test task")
    assert task_id > 0


def test_get_pending_tasks(queue):
    queue.enqueue_task("task-1", "First task")
    queue.enqueue_task("task-2", "Second task")
    tasks = queue.get_pending_tasks(10)
    assert len(tasks) == 2


def test_assign_task(queue):
    task_id = queue.enqueue_task("assign-test", "To be assigned")
    queue.register_worker("worker-1")
    result = queue.assign_task(task_id, "worker-1")
    assert result is True


def test_complete_task(queue):
    task_id = queue.enqueue_task("complete-test", "To be completed")
    queue.complete_task(task_id, "done")
    tasks = queue.get_pending_tasks(10)
    assert len(tasks) == 0


def test_register_worker(queue):
    queue.register_worker("worker-1")
    stats = queue.get_worker_stats()
    assert stats["online_workers"] >= 1
    assert "worker-1" in stats["active_workers"]


def test_worker_stats(queue):
    stats = queue.get_worker_stats()
    assert "online_workers" in stats
    assert "completed_tasks" in stats
    assert "active_workers" in stats
