from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import datasets

from manaita.preprocessor import register_preprocessor
from manaita.task_types import PreProcessorArgs

DatasetRow: TypeAlias = dict[str, Any]
Tokenizer: TypeAlias = Any


@register_preprocessor("casing", version="20250115_v0")
def apply_textcasing(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    casing: str = "lower",
    column: str = "text",
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _lowercasing(xs: Sequence[str], *, column: str) -> DatasetRow:
        return {column: [x.lower() for x in xs]}

    def _uppercasing(xs: Sequence[str], *, column: str) -> DatasetRow:
        return {column: [x.upper() for x in xs]}

    if casing == "upper":
        casing_fn = _uppercasing
    elif casing == "lower":
        casing_fn = _lowercasing
    else:
        raise ValueError("casing must be upper or lower")

    case_desc = "Uppercasing" if casing == "upper" else "Lowercasing"
    return ds.map(
        casing_fn,
        fn_kwargs={"column": column},
        input_columns=[column],
        batched=True,
        num_proc=args.num_proc,
        desc=f"{case_desc} [{ds.info.dataset_name}]",
    )


@register_preprocessor("audio", version="20250115_v0")
def apply_audio_preprocessing(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    resample: bool = False,
    remove_raw_audio: bool = True,
    override_source_rate: int | None = None,
    add_length_column: bool = True,
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _preprocess_audio(
        row: DatasetRow,
        *,
        audio_preprocessor: Any,
        add_length_column: bool = False,
    ) -> DatasetRow:
        out = audio_preprocessor(
            row["audio"]["array"], sampling_rate=row["audio"]["sampling_rate"]
        )
        row["input_values"] = out["input_values"][0]
        if add_length_column:
            row["input_value_length"] = out["input_values"][0].shape[0]
        return row

    def _resample(
        row: DatasetRow, *, target_rate: int, override_source_rate: int | None
    ) -> DatasetRow:
        source_rate = (
            row["audio"]["sampling_rate"]
            if override_source_rate is None
            else override_source_rate
        )
        audio_array = row["audio"]["array"]
        try:
            import librosa

            resampled = librosa.resample(
                audio_array,
                orig_sr=source_rate,
                target_sr=target_rate,
            )
        except ImportError:
            import numpy as np

            audio_array = np.asarray(audio_array)
            new_len = round(len(audio_array) * target_rate / source_rate)
            if new_len <= 0:
                resampled = np.asarray([], dtype=audio_array.dtype)
            else:
                resampled = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_len),
                    np.arange(len(audio_array)),
                    audio_array,
                )
        row["audio"] = {
            "array": resampled,
            "sampling_rate": target_rate,
        }
        return row

    if args.audio_preprocessor is None:
        raise ValueError("audio_preprocessor must be specified")

    num_proc = 1 if args.debug else args.num_proc

    if resample:
        target_rate = args.audio_preprocessor.sampling_rate
        ds = ds.map(
            _resample,
            num_proc=num_proc,
            fn_kwargs={
                "target_rate": target_rate,
                "override_source_rate": override_source_rate,
            },
            desc=f"Resampling audio [{ds.info.dataset_name}]",
        ).cast_column(
            "audio", datasets.Audio(sampling_rate=target_rate, num_channels=1)
        )

    ds = ds.map(
        _preprocess_audio,
        fn_kwargs={
            "audio_preprocessor": args.audio_preprocessor,
            "add_length_column": add_length_column,
        },
        num_proc=num_proc,
        desc=f"Applying AudioProcessor [{ds.info.dataset_name}]",
    )
    if remove_raw_audio:
        ds = ds.remove_columns(["audio"])
    return ds


@register_preprocessor("filter_by_column", version="20250115_v0")
def filter_by_column(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    column: str,
    invert: bool = False,
    equals_to: Any = None,
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _to_bool(x: Any, *, invert: bool) -> bool:
        return invert ^ bool(x)

    def _equals_to(x: Any, *, invert: bool, equals_to: Any) -> bool:
        return bool(invert ^ (x == equals_to))

    pred_fn: Callable[..., bool] = _to_bool
    fn_kwargs = {"invert": invert}
    if equals_to is not None:
        pred_fn = _equals_to
        fn_kwargs["equals_to"] = equals_to
    return ds.filter(
        pred_fn,
        fn_kwargs=fn_kwargs,
        input_columns=[column],
        num_proc=args.num_proc,
        desc=f"Filtering by `{column}` [{ds.info.dataset_name}]",
    )


@register_preprocessor("filter_by_shape", version="20250303_v0")
def filter_by_shape(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    column: str,
    axis: int,
    max_length: int,
    invert: bool = False,
    **kwargs: Any,
) -> datasets.Dataset:
    del kwargs

    def _check_shape_smaller_than(
        x: Any, *, invert: bool, axis: int, max_length: int
    ) -> bool:
        return invert ^ (
            len(x) <= max_length if axis == 0 else _shape(x)[axis] <= max_length
        )

    fn_kwargs = {"invert": invert, "max_length": max_length, "axis": axis}
    return ds.filter(
        _check_shape_smaller_than,
        fn_kwargs=fn_kwargs,
        input_columns=[column],
        num_proc=args.num_proc,
        desc=f"Filtering by shape of `{column}` [{ds.info.dataset_name}]",
    )


def _shape(x: Any) -> tuple[int, ...]:
    shape = getattr(x, "shape", None)
    if shape is not None:
        return tuple(cast(Sequence[int], shape))
    if isinstance(x, Sequence) and not isinstance(x, str):
        if len(x) == 0:
            return (0,)
        return (len(x), *_shape(x[0]))
    return ()


def tokenize_columns(
    ds: datasets.Dataset,
    columns: dict[str, str],
    tokenizer: Tokenizer,
    *,
    remove_text_columns: bool = False,
    tokenizer_kwargs: dict[str, Any] | None = None,
    num_proc: int | None = None,
    use_split_and_join: bool = False,
) -> datasets.Dataset:
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    def _tokenize_column(
        *xs_list: Sequence[str],
        tokenizer: Tokenizer,
        target_columns: list[str],
        tokenizer_kwargs: dict[str, Any],
    ) -> DatasetRow:
        return {
            target_column: tokenizer(xs, **tokenizer_kwargs)["input_ids"]
            for target_column, xs in zip(target_columns, xs_list)
        }

    ds_original = None
    if use_split_and_join:
        ds_original = ds
        if remove_text_columns:
            ds_original = ds_original.remove_columns(list(columns.keys()))
        remove_text_columns = True
        ds = ds.remove_columns([col for col in ds.column_names if col not in columns])

    ds = ds.map(
        _tokenize_column,
        input_columns=list(columns.keys()),
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "target_columns": list(columns.values()),
            "tokenizer_kwargs": tokenizer_kwargs,
        },
        desc=f"Tokenizing [{ds.info.dataset_name}]",
        num_proc=num_proc,
        remove_columns=list(columns.keys()) if remove_text_columns else [],
    )

    if use_split_and_join:
        if ds_original is None:
            raise AssertionError("original dataset must be saved for split-and-join")
        tokenized_ds, ds = ds, ds_original
        ds = ds.add_column("_joined", tokenized_ds)
        ds = ds.flatten().rename_columns(
            {"_joined." + col: col for col in columns.values()}
        )

    return ds
