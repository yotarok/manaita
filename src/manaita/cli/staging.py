import dataclasses
import importlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import transformers
import tyro
import yaml
from loguru import logger

import manaita.preprocessors  # noqa: F401  # registers built-in preprocessors
from manaita.staging import get_dataset_or_download_from_staging
from manaita.task_registry import register_task_loader
from manaita.task_types import PreProcessorArgs, TaskDef


@dataclasses.dataclass(frozen=True)
class Args:
    """Build and stage a dataset from YAML task definitions."""

    task_yaml: pathlib.Path
    staging_prefix: str
    import_modules: tuple[str, ...] = ()
    training: bool = True
    streaming: bool = False
    allow_build: bool = True
    stage_after_build: bool = True
    num_proc: int | None = None
    max_audio_samples: int | None = None
    max_label_tokens: int | None = None
    filter_columns: tuple[str, ...] | None = None
    debug: bool = False
    print_json: bool = True

    audio_preprocessor: str | None = None
    tokenizer: str | None = None


def _load_yaml(path: pathlib.Path) -> Mapping[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        raise ValueError(f"YAML file is empty: {path}")
    if not isinstance(loaded, Mapping):
        raise ValueError("task YAML must contain a mapping at the top level")
    return cast(Mapping[str, Any], loaded)


def _as_taskdef(value: Any, *, name: str) -> TaskDef:
    if not isinstance(value, Mapping):
        raise ValueError(f"task definition for {name!r} must be a mapping")

    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not all(
        isinstance(dataset, str) for dataset in datasets
    ):
        raise ValueError(f"task {name!r} must define `datasets` as a list of strings")

    preprocessors = value.get("preprocessors", [])
    if not isinstance(preprocessors, list):
        raise ValueError(f"task {name!r} must define `preprocessors` as a list")

    taskdef: dict[str, Any] = {
        "datasets": datasets,
        "preprocessors": preprocessors,
    }

    for key in (
        "preprocessor_args_override",
        "staging_opts",
        "eval_metric",
        "postprocessors",
        "pre_eval_normalizers",
        "generation_args_override",
    ):
        if key in value:
            taskdef[key] = value[key]

    return cast(TaskDef, taskdef)


def load_task_yaml(path: pathlib.Path) -> tuple[dict[str, TaskDef], dict[str, Any]]:
    raw = _load_yaml(path)
    preprocessor_args = raw.get("preprocessor_args", {})
    if preprocessor_args is None:
        preprocessor_args = {}
    if not isinstance(preprocessor_args, Mapping):
        raise ValueError("`preprocessor_args` must be a mapping when specified")

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, Mapping):
        raise ValueError("task YAML must define `tasks` as a mapping")
    tasks = {
        str(name): _as_taskdef(taskdef, name=str(name))
        for name, taskdef in tasks_raw.items()
    }

    return tasks, dict(preprocessor_args)


def _apply_cli_overrides(yaml_args: Mapping[str, Any], args: Args) -> dict[str, Any]:
    ret = dict(yaml_args)
    for key in (
        "num_proc",
        "max_audio_samples",
        "max_label_tokens",
        "filter_columns",
        "debug",
    ):
        value = getattr(args, key)
        if value is not None:
            ret[key] = value
    if args.filter_columns is not None:
        ret["filter_columns"] = tuple(args.filter_columns)
    ret["training"] = args.training
    return ret


def make_preprocessor_args(
    yaml_args: Mapping[str, Any], args: Args
) -> PreProcessorArgs:
    supported = {
        "max_audio_samples",
        "max_label_tokens",
        "num_proc",
        "training",
        "filter_columns",
        "debug",
    }
    values = _apply_cli_overrides(yaml_args, args)
    unknown = set(values) - supported
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported preprocessor_args field(s) in YAML: {names}")

    if args.audio_preprocessor is not None:
        values["audio_preprocessor"] = (
            transformers.AutoFeatureExtractor.from_pretrained(args.audio_preprocessor)
        )
    if args.tokenizer is not None:
        values["tokenizer"] = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    return PreProcessorArgs(**values)


def register_yaml_tasks(taskdefs: Mapping[str, TaskDef]) -> None:
    copied = dict(taskdefs)

    @register_task_loader
    def _yaml_task_loader() -> dict[str, TaskDef]:
        return copied


def run(args: Args) -> dict[str, Any]:
    for module_name in args.import_modules:
        importlib.import_module(module_name)

    taskdefs, yaml_preprocessor_args = load_task_yaml(args.task_yaml)
    register_yaml_tasks(taskdefs)

    preprocessor_args = make_preprocessor_args(yaml_preprocessor_args, args)
    staged_tasks = []
    for task_name in taskdefs:
        _, row_count = get_dataset_or_download_from_staging(
            task_name,
            preprocessor_args,
            staging_prefix=args.staging_prefix,
            training=args.training,
            streaming=args.streaming,
            allow_build=args.allow_build,
            stage_after_build=args.stage_after_build,
        )
        staged_tasks.append({"task": task_name, "row_count": row_count})

    result = {
        "tasks": staged_tasks,
        "task_count": len(staged_tasks),
        "staging_prefix": str(args.staging_prefix),
        "streaming": args.streaming,
        "training": args.training,
        "staged": args.stage_after_build,
    }
    if args.print_json:
        print(json.dumps(result, sort_keys=True))
    else:
        logger.info(f"Staged {len(staged_tasks)} task(s)")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = tyro.cli(Args, args=argv)
    run(args)


if __name__ == "__main__":
    main()
