import functools
from collections.abc import Callable, Mapping
from typing import Any

from manaita.task_types import PreProcessor

_PREPROCESSORS: dict[str, PreProcessor] = {}
_PREPROCESSOR_SIGNATURES: dict[str, str] = {}


def get_preprocessor(processordef: str | Mapping[str, Any]) -> PreProcessor:
    if isinstance(processordef, str):
        ret = _PREPROCESSORS.get(processordef)
        if ret is None:
            raise ValueError(f"Preprocessor ({processordef}) is not found")
        return ret
    funcname = processordef.get("op")
    if funcname is None:
        raise ValueError("preprocessor definition must have `op` field")
    ret = _PREPROCESSORS.get(funcname)
    if ret is None:
        raise ValueError(f"Preprocessor ({processordef}) is not found")
    kwargs = dict(processordef)
    kwargs.pop("op")
    return functools.partial(ret, **kwargs)


def get_preprocessor_signature(name: str) -> str | None:
    return _PREPROCESSOR_SIGNATURES.get(name)


def _register_preprocessor_impl(
    f: PreProcessor,
    name: str,
    signature: str | None = None,
    version: str | None = None,
) -> PreProcessor:
    if signature is not None:
        pass
    elif version is not None:
        signature = f"{name}@{version}"
    else:
        raise ValueError(
            f"Cannot determine the implementation signature of a preprocessor {name}"
        )

    _PREPROCESSORS[name] = f
    _PREPROCESSOR_SIGNATURES[name] = signature
    return f


def register_preprocessor(
    name: str, signature: str | None = None, version: str | None = None
) -> Callable[[PreProcessor], PreProcessor]:
    return functools.partial(
        _register_preprocessor_impl, name=name, signature=signature, version=version
    )
