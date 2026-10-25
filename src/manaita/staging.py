import json
import pathlib
import sys
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast, Callable

import cloudpathlib
import datasets
from loguru import logger

from manaita.task_registry import (
    get_dataset,
    get_iterable_dataset,
    get_task_descriptor,
    get_taskdef,
)
from manaita.task_types import PreProcessorArgs, strhash


def _generate_from_iterable_dataset(
    ds: datasets.IterableDataset,
    *,
    num_workers: int,
) -> datasets.Dataset:
    num_shards = ds.num_shards

    def gen(shard_ids):
        for i in shard_ids:
            logger.info(f"generating {i}-th shard (num_shards={num_shards})...")
            yield from ds.shard(num_shards=num_shards, index=i)
            logger.info(f"generating {i}-th shard (num_shards={num_shards}) completed.")

    return cast(
        datasets.Dataset,
        datasets.Dataset.from_generator(
            gen,
            gen_kwargs={"shard_ids": list(range(num_shards))},
            num_proc=min(num_shards, num_workers),
            writer_batch_size=32,
        ),
    )


def _save_sharded_parquet(
    ds: datasets.Dataset | datasets.IterableDataset,
    *,
    num_shards: int,
    staging_path: cloudpathlib.CloudPath,
    num_workers: int,
) -> datasets.Dataset:
    if isinstance(ds, datasets.IterableDataset):
        ds = _generate_from_iterable_dataset(ds, num_workers=num_workers)

    def _write_shard(i: int) -> None:
        shard = ds.shard(num_shards=num_shards, index=i)
        shard.to_parquet(
            str(staging_path / f"shard-{i:05d}-of-{num_shards:05d}.parquet")
        )

    staging_path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(num_workers, num_shards)) as executor:
        list(executor.map(_write_shard, range(num_shards)))

    return ds


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


def _without_lock(f: Callable[..., Any]) -> Callable[..., Any]:
    return f


lock = threading.Lock()


def _with_global_lock(f: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped_f(*args, **kwargs):
        with lock:
            ret = f(*args, **kwargs)
        return ret

    return wrapped_f


def get_dataset_or_download_from_staging(
    task: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    staging_prefix: str | None,
    training: bool = True,
    streaming: bool = True,
    allow_build: bool = True,
    stage_after_build: bool = True,
    forbid_parallel_build: bool = True,
) -> tuple[datasets.Dataset | datasets.IterableDataset, int]:
    taskdef = get_taskdef(task)
    staging_opts = taskdef.get("staging_opts", {})
    num_shards = staging_opts.get("num_shards", 20)
    if num_shards <= 0 or num_shards > 99_999:
        raise ValueError(f"{num_shards=} must be 0 < num_shards < 100000.")

    maybe_with_lock = _with_global_lock if forbid_parallel_build else _without_lock

    if preproc_args is None:
        preproc_args = PreProcessorArgs()

    if staging_prefix is None:
        logger.info(
            f"Staging prefix is not given. Starting data preparation (streaming={streaming})."
        )
        if streaming:
            # No staging + iterable does not need a critical section
            ds = get_iterable_dataset(task, preproc_args, training=training)
            row_count = -1
        else:
            # No staging + non-iterable needs a critical section
            ds = maybe_with_lock(get_dataset)(task, preproc_args, training=training)
            row_count = len(ds)
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

    logger.info(
        f"Prepared data is not found. Starting data preparation. (streaming={streaming})"
    )
    if streaming:
        # computation does not happen here, so no lock required
        ds = get_iterable_dataset(task, preproc_args, training=training)
        row_count = -1
    else:
        ds = maybe_with_lock(get_dataset)(task, preproc_args, training=training)
        row_count = len(ds)

    if stage_after_build:
        logger.info(
            f"Storing the prepared dataset to {staging_path}. (#shards={num_shards})"
        )
        generated_ds = _save_sharded_parquet(
            ds,
            num_shards=num_shards,
            staging_path=staging_path,
            num_workers=preproc_args.num_proc,
        )
        _write_metadata(staging_path, descriptor=task_descriptor, ds=generated_ds)
        if streaming:
            ds = maybe_with_lock(
                lambda: generated_ds.to_iterable_dataset(num_shards=num_shards)
            )()

    if preproc_args.cleanup_cache_after_staging:
        ds.cleanup_cache_files()

    if streaming:
        return ds, -1
    else:
        row_count = len(ds)
        return ds, row_count


def get_multi_datasets_or_download_from_staging(
    tasks: Iterable[str],
    preproc_args: PreProcessorArgs | None = None,
    *,
    staging_prefix: str | None,
    training: bool = True,
    streaming: bool = True,
    allow_build: bool = True,
    stage_after_build: bool = True,
) -> list[tuple[datasets.Dataset | datasets.IterableDataset, int]]:
    if preproc_args is None:
        preproc_args = PreProcessorArgs()

    ret = []
    logger.info("Resolving multiple datasets in parallel: {}", str(tasks))
    with ThreadPoolExecutor(max_workers=preproc_args.num_proc) as executor:
        futures = [
            executor.submit(
                get_dataset_or_download_from_staging,
                task,
                preproc_args,
                staging_prefix=staging_prefix,
                training=training,
                streaming=streaming,
                allow_build=allow_build,
                stage_after_build=stage_after_build,
                forbid_parallel_build=True,
            )
            for task in tasks
        ]
        for future in futures:
            ret.append(future.result())
    return ret
