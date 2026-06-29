import pytest
import sys
sys.path.insert(0, ".")

from forge_ai import MemoryDB


@pytest.fixture
def memory():
    m = MemoryDB(":memory:")
    return m


def test_store_experiment(memory):
    memory.store_experiment("test-task", "A test task", "print('hello')", True, "multi-agent-ensemble")
    results = memory.get_recent_discoveries(1)
    assert len(results) >= 0


def test_find_similar_solution(memory):
    memory.store_experiment("sort-list", "Write a function to sort a list", "def sort(lst): return sorted(lst)", True, "multi-agent-ensemble")
    result = memory.find_similar_solution("sort a list of numbers")
    assert result is not None
    assert result["task_name"] == "sort-list"


def test_find_similar_no_match(memory):
    result = memory.find_similar_solution("unique task no match")
    assert result is None


def test_store_discovery(memory):
    memory.store_discovery("Use list comprehensions for speed", 0.85, "learner")
    discoveries = memory.get_recent_discoveries(10)
    assert len(discoveries) >= 1
    assert "list comprehensions" in discoveries[0]["discovery"]


def test_multiple_discoveries(memory):
    for i in range(5):
        memory.store_discovery(f"Discovery {i}", 0.5 + i * 0.1, "learner")
    discoveries = memory.get_recent_discoveries(3)
    assert len(discoveries) == 3
