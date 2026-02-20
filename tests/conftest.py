import datasets
import pytest

from manaita import dataset_uri


@pytest.fixture
def fake_dataset():
    return datasets.Dataset.from_dict({"n": list(range(5))})


def _load_fake_dataset(
    uri: dataset_uri.DatasetUri, streaming: bool = False, num_proc: int | None = None
) -> datasets.Dataset | datasets.IterableDataset:
    del num_proc
    ds = fake_dataset()
    if streaming:
        return ds.to_iterable_dataset(2)
    return ds


@pytest.fixture
def with_fake_data_source():
    dataset_uri.register_data_source("fake", allow_override=True)(_load_fake_dataset)
    yield None
