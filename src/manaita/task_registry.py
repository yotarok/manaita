import copy
import dataclasses
import functools
from collections.abc import Iterable
from typing import Any, TypeVar

import datasets
from loguru import logger
import multiprocess

from manaita.dataset import DatasetProcessor, concatenate_dataset_processors
from manaita.dataset_uri import DatasetUri
from manaita.preprocessor import get_preprocessor, get_preprocessor_signature
from manaita.task_types import (
    PreProcessorArgs,
    TaskDef,
    TaskLoader,
    TrainingTaskDef,
    PreProcessorDef,
)

_TASK_LOADERS: tuple[TaskLoader, ...] = ()


@functools.cache
def _run_task_loaders(loaders: tuple[TaskLoader, ...]) -> dict[str, TaskDef]:
    ret: dict[str, TaskDef] = {}
    for loader in loaders:
        loaded = loader()
        duplicated = set(ret).intersection(loaded)
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(f"duplicated task definition(s): {names}")
        ret.update(loaded)
    return ret


def _taskdefs() -> dict[str, TaskDef]:
    return _run_task_loaders(_TASK_LOADERS)


def register_task_loader(f: TaskLoader) -> TaskLoader:
    global _TASK_LOADERS
    _TASK_LOADERS = (*_TASK_LOADERS, f)
    return f


def get_all_task_names() -> tuple[str, ...]:
    return tuple(sorted(_taskdefs()))


def get_taskdef(task_name: str) -> TaskDef:
    taskdef = _taskdefs().get(task_name)
    if taskdef is None:
        raise ValueError(f"task {task_name=} is not defined.")
    return copy.deepcopy(taskdef)


def _apply_1d_length_filter(
    ds: DatasetProcessor,
    column: str,
    max_len: int,
    *,
    dim: int = 0,
    num_proc: int | None = None,
    desc: str | None = None,
) -> DatasetProcessor:
    first_value = ds.peek_first()
    if first_value is None:
        return ds

    if isinstance(first_value[column], list):
        return ds.filter(
            lambda x: len(x) <= max_len,
            input_columns=[column],
            num_proc=num_proc,
            desc=desc,
        )
    return ds.filter(
        lambda x: x.shape[dim] <= max_len,
        input_columns=[column],
        num_proc=num_proc,
        desc=desc,
    )


def _preprocess_dataset(
    subdatasets: Iterable[DatasetProcessor],
    taskdef: TrainingTaskDef,
    preproc_args: PreProcessorArgs,
) -> list[DatasetProcessor]:
    if "preprocessor_args_override" in taskdef:
        preproc_args = dataclasses.replace(
            preproc_args, **taskdef["preprocessor_args_override"]
        )

    ret: list[datasets.Dataset] = list(subdatasets)
    for preprocdef in taskdef["preprocessors"]:
        preproc = get_preprocessor(preprocdef)
        ret = [preproc(ds, preproc_args) for ds in ret]

    if preproc_args.max_label_tokens is not None:
        ret = [
            _apply_1d_length_filter(
                ds,
                "label_ids",
                max_len=preproc_args.max_label_tokens,
                num_proc=preproc_args.num_proc,
                desc=f"Filtering by length of 'label_ids'",
            )
            for ds in ret
        ]

    if preproc_args.max_audio_samples is not None:
        max_len = preproc_args.max_audio_samples
        ret = [
            ds.filter(
                lambda x: x <= max_len,
                input_columns=["input_value_length"],
                num_proc=preproc_args.num_proc,
                desc=f"Filtering by length of 'input_values'",
            )
            for ds in ret
        ]

    if preproc_args.filter_columns is not None:
        ret = [
            ds.remove_columns(
                [
                    name
                    for name in ds.column_names
                    if name not in preproc_args.filter_columns
                ]
            )
            for ds in ret
        ]

    return ret


def _normalize_preproc_args(
    preproc_args: PreProcessorArgs | None, *, training: bool
) -> PreProcessorArgs:
    if preproc_args is None:
        preproc_args = PreProcessorArgs()
    if preproc_args.training is not None:
        logger.info(
            "`training` flag in `PreProcessorArgs` is set manually. This setting is used."
        )
        return preproc_args
    return dataclasses.replace(preproc_args, training=training)


def get_task_descriptor(
    task_name: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    training: bool = True,
) -> dict[str, Any]:
    taskdef = get_taskdef(task_name)
    taskdef.pop("staging_opts", None)

    taskdef["datasets"] = list(sorted(taskdef["datasets"]))
    normalized_args = _normalize_preproc_args(preproc_args, training=training)
    if "preprocessor_args_override" in taskdef:
        normalized_args = dataclasses.replace(
            normalized_args, **taskdef["preprocessor_args_override"]
        )

    preprocessor_defs = []
    for ppdef in taskdef["preprocessors"]:
        ppdef: PreProcessorDef = (
            PreProcessorDef.from_str(ppdef)
            if isinstance(ppdef, str)
            else PreProcessorDef(ppdef)
        )
        op = ppdef["op"]
        if not isinstance(op, str):
            raise ValueError(f"preprocessor op must be a string: {ppdef}")
        ppdef["signature"] = get_preprocessor_signature(op)
        preprocessor_defs.append(ppdef)

    return {
        "datasets": taskdef["datasets"],
        "preprocessors": preprocessor_defs,
        "preprocessor_args": normalized_args.descriptor,
    }


def _load_dataset(
    dsuri: str,
    *,
    streaming: bool = False,
    num_proc: int | None = None,
) -> datasets.Dataset | datasets.IterableDataset:
    return DatasetUri.from_string(dsuri).load_dataset(
        loader_kwargs={"streaming": streaming, "num_proc": num_proc}
    )


def get_dataset_processor(
    task_name: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    streaming: bool,
    training: bool = True,
) -> DatasetProcessor:
    taskdef = get_taskdef(task_name)
    normalized_args = _normalize_preproc_args(preproc_args, training=training)

    subdatasets = [
        DatasetProcessor(
            _load_dataset(dsuri, num_proc=normalized_args.num_proc, streaming=streaming)
        )
        for dsuri in taskdef["datasets"]
    ]

    previous_start_method = multiprocess.get_start_method(allow_none=True)
    try:
        if normalized_args.multiprocess_start_method:
            multiprocess.set_start_method(normalized_args.multiprocess_start_method)
        processed = _preprocess_dataset(subdatasets, taskdef, normalized_args)
    finally:
        if previous_start_method is not None:
            multiprocess.set_start_method(previous_start_method)

    ret = concatenate_dataset_processors(processed)

    logger.info(f"Preparation of {task_name} finished.")
    return ret


def get_dataset(
    task_name: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    training: bool = True,
) -> datasets.Dataset:
    return get_dataset_processor(
        task_name, preproc_args, streaming=False, training=training
    ).as_dataset()


def get_iterable_dataset(
    task_name: str,
    preproc_args: PreProcessorArgs | None = None,
    *,
    training: bool = True,
) -> datasets.IterableDataset:
    return get_dataset_processor(
        task_name, preproc_args, streaming=True, training=training
    ).as_iterable_dataset()
