from manaita import staging, task_registry
from manaita.task_types import TaskDef


def test_staging_uses_small_local_dataset(with_fake_data_source, tmp_path):
    @task_registry.register_task_loader
    def _tasks() -> dict[str, TaskDef]:
        return {
            "staging_task": {
                "datasets": ["fake:noname"],
                "preprocessors": [],
                "staging_opts": {"num_shards": 2},
            }
        }

    ds, row_count = staging.get_dataset_or_download_from_staging(
        "staging_task",
        staging_prefix=tmp_path,
        streaming=False,
        training=False,
    )

    assert row_count == 5
    assert len(ds) == 5

    staged_ds, staged_row_count = staging.get_dataset_or_download_from_staging(
        "staging_task",
        staging_prefix=tmp_path,
        streaming=False,
        training=False,
        allow_build=False,
    )

    assert staged_row_count == 5
    assert len(staged_ds) == 5
