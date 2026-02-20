import json
import pathlib
import sys
from typing import Any, cast

import cloudpathlib
import datasets
from loguru import logger

from manaita.task_registry import get_dataset, get_task_descriptor, get_taskdef
from manaita.task_types import PreProcessorArgs, strhash


def _save_sharded_parquet(
    ds: datasets.Dataset,
    *,
    num_shards: int,
    staging_path: cloudpathlib.CloudPath,
) -> None:
    staging_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_shards):
        ds.shard(num_shards=num_shards, index=i).to_parquet(
            str(staging_path / f"shard-{i:05d}-of-{num_shards:05d}.parquet")
        )


def _load_staged_dataset(
    staging_path: cloudpathlib.CloudPath,
    *,
    streaming: bool,
    num_proc: int | None,
) -> datasets.Dataset | datasets.IterableDataset:
    ds = datasets.load_dataset(
        "parquet",
        data_files=str(staging_path / "shard-*-of-*.parquet"),
        num_proc=None if streaming else num_proc,
        split=datasets.Split.TRAIN,
        streaming=streaming,
        keep_in_memory=False,
    )
    if not isinstance(ds, (datasets.Dataset, datasets.IterableDataset)):
        raise TypeError(f"unexpected staged dataset type: {type(ds)}")
    return ds


def _read_info(staging_path: pathlib.Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((staging_path / "info.json").read_text()))


def _write_metadata(
    staging_path: pathlib.Path,
    *,
    descriptor: str,
    ds: datasets.Dataset,
) -> None:
    (staging_path / "descriptor.json").write_text(descriptor)
    (staging_path / "info.json").write_text(
        json.dumps(
            {
                "num_example": len(ds),
                "features": ds.features.to_dict(),
                "provenance": {
                    "argv": sys.argv,
                },
            },
        )
    )


def get_dataset_or_download_from_staging(
    task: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    staging_prefix: str | None,
    training: bool = True,
    streaming: bool = True,
    allow_build: bool = True,
    stage_after_build: bool = True,
) -> tuple[datasets.Dataset | datasets.IterableDataset, int]:
    taskdef = get_taskdef(task)
    staging_opts = taskdef.get("staging_opts", {})
    num_shards = staging_opts.get("num_shards", 20)
    if num_shards <= 0 or num_shards > 99_999:
        raise ValueError(f"{num_shards=} must be 0 < num_shards < 100000.")

    if preproc_args is None:
        preproc_args = PreProcessorArgs()

    if staging_prefix is None:
        logger.info("Staging prefix is not given. Starting data preparation.")
        ds = get_dataset(task, preproc_args, training=training)
        row_count = len(ds)
        if streaming:
            return ds.to_iterable_dataset(num_shards=num_shards), row_count
        return ds, row_count

    task_descriptor = json.dumps(
        get_task_descriptor(task, preproc_args, training=training),
        sort_keys=True,
    )
    logger.info(f"{task_descriptor=}")
    task_hash = strhash(task_descriptor)[:16]
    staging_path = cloudpathlib.AnyPath(staging_prefix) / task / task_hash

    logger.info(f"Staging for {task} is {staging_path}")
    if (staging_path / "info.json").exists():
        logger.info("Prepared data found in the staging location.")
        info = _read_info(staging_path)
        ds = _load_staged_dataset(
            staging_path,
            streaming=streaming,
            num_proc=preproc_args.num_proc,
        )
        row_count = int(info["num_example"]) if streaming else len(ds)
        return ds, row_count

    if not allow_build:
        raise ValueError(
            f"Dataset for {task} is not found in {staging_path}. "
            "On-demand data preparation is not allowed."
        )

    logger.info("Prepared data is not found. Starting data preparation.")
    ds = get_dataset(task, preproc_args, training=training)

    if stage_after_build:
        logger.info(
            f"Storing the prepared dataset to {staging_path}. (#shards={num_shards})"
        )
        _save_sharded_parquet(ds, num_shards=num_shards, staging_path=staging_path)
        _write_metadata(staging_path, descriptor=task_descriptor, ds=ds)

    row_count = len(ds)
    if streaming:
        return ds.to_iterable_dataset(num_shards=num_shards), row_count
    return ds, row_count
