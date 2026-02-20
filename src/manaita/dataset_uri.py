import dataclasses
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import aiohttp
import datasets
from loguru import logger


class DataSource(Protocol):
    """Protocol for custom data loaders."""

    def __call__(
        self, uri: "DatasetUri", /, streaming: bool = False, num_proc: int | None = None
    ) -> datasets.Dataset | datasets.IterableDataset: ...


@dataclasses.dataclass
class DatasetUri:
    """Utility class for specifying a dataset from, e.g. command-line argument.

    A dataset URI has the following form.

     - full: "[SOURCE]:[NAME]:[SPLIT]?[PARAMTERS]"
     - w/o source: "[NAME]:[SPLIT]"
     - w/o split: "[NAME]"

    [SOURCE] and [NAME] does not include ':' or '?', and [SPLIT] does not include '?'
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
        param_str = ""
        if "?" in s:
            s, param_str = s.split("?", maxsplit=1)

        components = s.split(":", maxsplit=2)

        source = None
        name = None
        split = None
        if len(components) == 3:
            source, name, split = components
        elif len(components) == 2:
            name, split = components
        elif len(components) == 1:
            (name,) = components
        else:
            raise ValueError("`DatasetUri.from_string` is called with an empty string")

        parameters = []
        if param_str:
            for kvpair in param_str.split("&"):
                k, v = kvpair.split("=", maxsplit=1)
                parameters.append((k, v))

        return cls(source, name, split, parameters)


_HFHUB_LOAD_DATASET_OVERRIDABLE_PARAMETERS = {"revision", "data_dir"}


def load_dataset_from_hugging_face_hub(
    uri: DatasetUri, streaming: bool = False, num_proc: int | None = None
) -> datasets.Dataset | datasets.IterableDataset:
    hf_user_name, hf_dataset_name, hf_config_name = (uri.name.split("/") + [None])[:3]

    load_opts = {}
    for k, v in uri.parameters:
        if k in _HFHUB_LOAD_DATASET_OVERRIDABLE_PARAMETERS:
            load_opts[k] = v
        else:
            warnings.warn(f"Parameter {k} is specified in {uri}, but ignored.")

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


_DATA_SOURCES: dict[str, DataSource] = {
    "": load_dataset_from_hugging_face_hub,
    "hfhub": load_dataset_from_hugging_face_hub,
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
