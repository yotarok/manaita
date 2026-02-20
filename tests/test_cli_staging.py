import datasets

from manaita import dataset_uri
from manaita.cli import staging as staging_cli


def test_cli_staging_from_yaml(tmp_path, capsys):
    def _load_cli_fake_dataset(
        uri: dataset_uri.DatasetUri,
        streaming: bool = False,
        num_proc: int | None = None,
    ) -> datasets.Dataset | datasets.IterableDataset:
        del uri, num_proc
        ds = datasets.Dataset.from_dict(
            {
                "text": ["HELLO", "WORLD"],
                "keep": [True, False],
            }
        )
        if streaming:
            return ds.to_iterable_dataset(1)
        return ds

    dataset_uri.register_data_source("cli_fake", allow_override=True)(
        _load_cli_fake_dataset
    )

    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        """
tasks:
  cli_task_a:
    datasets:
      - cli_fake:anything:train
    preprocessors:
      - op: casing
        casing: lower
      - op: filter_by_column
        column: keep
    staging_opts:
      num_shards: 1
  cli_task_b:
    datasets:
      - cli_fake:anything:train
    preprocessors:
      - op: casing
        casing: lower
    staging_opts:
      num_shards: 1
preprocessor_args:
  num_proc:
"""
    )

    result = staging_cli.run(
        staging_cli.Args(
            task_yaml=task_yaml,
            staging_prefix=tmp_path / "staging",
            training=False,
        )
    )

    captured = capsys.readouterr()

    assert result["task_count"] == 2
    assert result["tasks"] == [
        {"task": "cli_task_a", "row_count": 1},
        {"task": "cli_task_b", "row_count": 2},
    ]
    assert '"row_count": 1' in captured.out
    assert '"row_count": 2' in captured.out
    assert list((tmp_path / "staging" / "cli_task_a").glob("*/info.json"))
    assert list((tmp_path / "staging" / "cli_task_b").glob("*/info.json"))


def test_cli_loads_tasks_mapping(tmp_path):
    task_yaml = tmp_path / "tasks.yaml"
    task_yaml.write_text(
        """
tasks:
  a:
    datasets: ["x:a"]
    preprocessors: []
  b:
    datasets: ["x:b"]
    preprocessors: []
"""
    )

    tasks, preprocessor_args = staging_cli.load_task_yaml(task_yaml)

    assert set(tasks) == {"a", "b"}
    assert preprocessor_args == {}


def test_cli_rejects_single_task_yaml(tmp_path):
    task_yaml = tmp_path / "single.yaml"
    task_yaml.write_text(
        """
name: unsupported_single_task
datasets: ["x:a"]
preprocessors: []
"""
    )

    try:
        staging_cli.load_task_yaml(task_yaml)
    except ValueError as exc:
        assert "`tasks`" in str(exc)
    else:
        raise AssertionError("single-task YAML should be rejected")


def test_cli_main_uses_tyro_argv(tmp_path, capsys):
    def _load_cli_main_dataset(
        uri: dataset_uri.DatasetUri,
        streaming: bool = False,
        num_proc: int | None = None,
    ) -> datasets.Dataset | datasets.IterableDataset:
        del uri, num_proc
        ds = datasets.Dataset.from_dict({"text": ["ONE"]})
        if streaming:
            return ds.to_iterable_dataset(1)
        return ds

    dataset_uri.register_data_source("cli_main_fake", allow_override=True)(
        _load_cli_main_dataset
    )

    task_yaml = tmp_path / "task-main.yaml"
    task_yaml.write_text(
        """
tasks:
  cli_main_task:
    datasets:
      - cli_main_fake:anything:train
    preprocessors:
      - op: casing
        casing: lower
    staging_opts:
      num_shards: 1
"""
    )

    staging_cli.main(
        [
            "--task-yaml",
            str(task_yaml),
            "--staging-prefix",
            str(tmp_path / "staging"),
            "--no-training",
        ]
    )

    captured = capsys.readouterr()

    assert '"task_count": 1' in captured.out
    assert '"task": "cli_main_task"' in captured.out
    assert '"row_count": 1' in captured.out
