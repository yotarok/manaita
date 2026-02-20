import datasets
import pytest

from manaita import preprocessor
from manaita.task_types import PreProcessorArgs


def test_registration():
    @preprocessor.register_preprocessor("f", version="v0")
    def _f(ds: datasets.Dataset) -> datasets.Dataset:
        return ds

    preprocessor.get_preprocessor("f")

    with pytest.raises(ValueError):
        preprocessor.get_preprocessor("unknown_preproc")


def test_training():
    @preprocessor.register_preprocessor("add_one", version="v0")
    def _add_one(ds: datasets.Dataset, args: PreProcessorArgs) -> datasets.Dataset:
        return ds.map(
            lambda xs: {"y": [x + 1 for x in xs]},
            input_columns=["x"],
            batched=True,
            num_proc=args.num_proc,
        )

    ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})
    proc = preprocessor.get_preprocessor("add_one")

    out = proc(ds, PreProcessorArgs(num_proc=None))

    assert out["y"] == [2, 3, 4]
