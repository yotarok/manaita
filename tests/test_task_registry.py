import pytest

from manaita import task_registry
from manaita.task_types import TaskDef


@pytest.mark.usefixtures("with_fake_data_source")
def test_task_registry():
    @task_registry.register_task_loader
    def _tasks() -> dict[str, TaskDef]:
        return {
            "task1": {
                "datasets": ["fake:noname"],
                "preprocessors": [],
            }
        }

    taskdef = task_registry.get_taskdef("task1")

    assert "datasets" in taskdef
    assert "task1" in task_registry.get_all_task_names()

    ds = task_registry.get_dataset("task1", shuffle=False)

    assert len(ds) == 5
    assert ds["n"] == [0, 1, 2, 3, 4]

    with pytest.raises(ValueError):
        task_registry.get_taskdef("task0")
