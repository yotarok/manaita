import pytest

import manaita
from manaita import TaskDef


@manaita.register_preprocessor("casing", version="v0_20260324")
def apply_textcasing(
    ds: datasets.Dataset,
    args: PreProcessorArgs,
    casing: str = "lower",
    column: str = "text",
) -> datasets.Dataset:
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


@manaita.register_task_loader
def tasks() -> dict[str, TaskDef]:
    return {
        "dummy_train": {
            "datasets": ["hfhub:hf-internal-testing/librispeech_asr_dummy"],
            "preprocessors": [
                {
                    "op": "textcasing",
                    "casing": "lower",
                },
            ],
        },
    }


@pytest.mark.integration
def test_timit_preparation():
    manaita.get_dataset("dummy_train")

    pass
