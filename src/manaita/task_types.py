import abc
import dataclasses
import hashlib
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, Protocol, TypeAlias, TypedDict

import datasets


class PreEvalNormalizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ref: str, gen: str, **kwargs: Any) -> tuple[str, str]: ...


def strhash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class PreProcessorArgs:
    tokenizer: Any | None = None
    audio_preprocessor: Any | None = None
    max_audio_samples: int | None = None
    max_label_tokens: int | None = None
    num_proc: int | None = None
    training: bool | None = None
    filter_columns: Sequence[str] | None = None
    debug: bool = False

    @property
    def descriptor(self) -> dict[str, Any]:
        tokenizer_hash = ""
        if self.tokenizer is not None:
            backend_tokenizer = getattr(self.tokenizer, "backend_tokenizer", None)
            if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
                tokenizer_hash = strhash(backend_tokenizer.to_str())
            else:
                tokenizer_hash = repr(self.tokenizer)

        audio_preprocessor_desc: Any = ""
        if self.audio_preprocessor is not None:
            if hasattr(self.audio_preprocessor, "to_dict"):
                audio_preprocessor_desc = self.audio_preprocessor.to_dict()
            else:
                audio_preprocessor_desc = repr(self.audio_preprocessor)

        return {
            "tokenizer": tokenizer_hash,
            "audio_preprocessor": audio_preprocessor_desc,
            "max_audio_samples": self.max_audio_samples,
            "max_label_tokens": self.max_label_tokens,
            "num_proc": self.num_proc,
            "training": self.training,
            "filter_columns": None
            if self.filter_columns is None
            else list(sorted(self.filter_columns)),
            "debug": self.debug,
        }


class PreProcessor(Protocol):
    def __call__(
        self, ds: datasets.Dataset, args: PreProcessorArgs, *args_: Any, **kwargs: Any
    ) -> datasets.Dataset: ...


class StagingOpts(TypedDict, total=False):
    num_shards: int


class TrainingTaskDef(TypedDict):
    datasets: list[str]
    preprocessors: list[str | dict[str, Any]]
    preprocessor_args_override: NotRequired[dict[str, Any]]
    staging_opts: NotRequired[StagingOpts]


class EvaluationTaskDef(TrainingTaskDef):
    eval_metric: str
    postprocessors: NotRequired[list[str | dict[str, Any]]]
    pre_eval_normalizers: NotRequired[list[str | dict[str, Any]]]
    generation_args_override: NotRequired[dict[str, Any]]


TaskDef: TypeAlias = TrainingTaskDef | EvaluationTaskDef
TaskLoader: TypeAlias = Callable[[], dict[str, TaskDef]]
