import dataclasses
import urllib.parse
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol

import aiohttp
import datasets
from loguru import logger


class DataSource(Protocol):
    """Protocol for custom data loaders."""

    def __call__(
        self, uri: "DatasetUri", /, streaming: bool = False, num_proc: int | None = None
    ) -> datasets.Dataset | datasets.IterableDataset: ...


def _extract_parameters(s: str) -> tuple[list[tuple[str, str]], str]:
    s, param_str, *_ = s.split("?", maxsplit=1) + [""]
    parameters = []
    if param_str:
        for kvpair in param_str.split("&"):
            k, v = kvpair.split("=", maxsplit=1)
            k = urllib.parse.unquote(k)
            v = urllib.parse.unquote(v)
            parameters.append((k, v))
    return parameters, s


@dataclasses.dataclass
class DatasetUri:
    """Utility class for specifying a dataset from, e.g. command-line argument.

    A dataset URI has the following form.
     - "hfhub:[NAME]:[SPLIT]?[PARAMETERS]"
     - "parquet:gs://[GLOB]?[PARAMETERS]"

    where all components (NAME, SPLIT, GLOB, PARAMETERS) are URL-encoded.
    """

    source: str | None
    name: str
    split: str | None
    parameters: Sequence[tuple[str, str]]

    def load_dataset(
        self, loader_kwargs: dict[str, Any] | None = None
    ) -> datasets.Dataset | datasets.IterableDataset:
        loader = (
            _DATA_SOURCES[""] if self.source is None else _DATA_SOURCES[self.source]
        )
        loader_kwargs = {} if loader_kwargs is None else loader_kwargs
        return loader(self, **loader_kwargs)

    @classmethod
    def from_string(cls, s: str) -> "DatasetUri":
        source, rest = s.split(":", maxsplit=1)
        if source == "hfhub":
            return cls.from_string_hfhub(rest)
        elif source == "parquet":
            return cls.from_string_parquet(rest)
        else:
            raise ValueError(f"Unknow data-source specified input={s} source={source}")

    @classmethod
    def from_string_hfhub(cls, s: str) -> "DatasetUri":
        parameters, s = _extract_parameters(s)
        source = "hfhub"
        name, split, *_ = s.split(":", maxsplit=1) + [None]
        name = urllib.parse.unquote(name)
        split = urllib.parse.unquote(split)
        return cls(source, name, split, parameters)

    @classmethod
    def from_string_parquet(cls, s: str) -> "DatasetUri":
        parameters, name = _extract_parameters(s)
        name = urllib.parse.unquote(name)
        source = "parquet"
        return cls(source, name, None, parameters)


_HFHUB_LOAD_DATASET_OVERRIDABLE_PARAMETERS = {"revision", "data_dir"}


def _extract_load_opts(parameters: dict[str, str]) -> dict[str, Any]:
    load_opts = {}
    to_be_removed = []
    for k, v in parameters:
        if k in _HFHUB_LOAD_DATASET_OVERRIDABLE_PARAMETERS:
            load_opts[k] = v
            to_be_removed.append(k)
        else:
            warnings.warn(f"Parameter {k} is specified in {uri}, but ignored.")

    parameters = parameters.copy()
    for k in to_be_removed:
        parameters.pop(k)
    return load_opts, parameters


def load_dataset_from_hugging_face_hub(
    uri: DatasetUri, streaming: bool = False, num_proc: int | None = None
) -> datasets.Dataset | datasets.IterableDataset:
    hf_user_name, hf_dataset_name, hf_config_name = (uri.name.split("/") + [None])[:3]

    load_opts, _other_parameters = _extract_load_opts(dict(uri.parameters))
    logger.info(f"Loading dataset [{uri.name}:{uri.split}] from hfhub. ({load_opts=})")
    dataset = datasets.load_dataset(
        f"{hf_user_name}/{hf_dataset_name}",
        hf_config_name,
        split=uri.split,
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
        streaming=streaming,
        num_proc=num_proc,
        **load_opts,
    )

    assert isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
    return dataset


def load_dataset_from_parquet_files(
    uri: DatasetUri, streaming: bool = False, num_proc: int | None = None
) -> datasets.Dataset | datasets.IterableDataset:
    load_opts, _other_parameters = _extract_load_opts(dict(uri.parameters))
    logger.info(f"Loading dataset parquet glob {uri.name}. ({load_opts=})")

    dss = datasets.load_dataset(
        "parquet",
        data_files=uri.name,
        split=uri.split,
        num_proc=num_proc,
        streaming=streaming,
        **load_opts,
    )
    ds = dss[uri.split or "train"]
    assert isinstance(ds, (datasets.Dataset, datasets.IterableDataset))
    return ds


_DATA_SOURCES: dict[str, DataSource] = {
    "": load_dataset_from_hugging_face_hub,
    "hfhub": load_dataset_from_hugging_face_hub,
    "parquet": load_dataset_from_parquet_files,
}


def register_data_source(
    name: str, *, override_default: bool = False, allow_override: bool = False
) -> Callable[[DataSource], DataSource]:
    def decorator(src: DataSource) -> DataSource:
        if not allow_override and name in _DATA_SOURCES:
            raise ValueError(
                f"data source with name='{name}' is already registered. use different name"
            )
        _DATA_SOURCES[name] = src
        if override_default:
            _DATA_SOURCES[""] = src
        return src

    return decorator
