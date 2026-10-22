import dataclasses

import datasets
from datasets import Dataset, IterableDataset


def _convert_to_non_iterable(ids: IterableDataset) -> Dataset:
    def gen():
        yield from ids

    ds = Dataset.from_generator(gen, features=ids.features)
    assert isinstance(ds, Dataset)
    return ds


@dataclasses.dataclass
class DatasetProcessContext:
    num_proc: int | None = None
    task_name: str = ""


class DatasetProcessor:
    """Datatype for enabling type-agnostic perprocessor."""

    def __init__(self, inner: Dataset | IterableDataset) -> None:
        self.dataset = None
        self.iterable_dataset = None

        if isinstance(inner, Dataset):
            self.dataset = inner
        elif isinstance(inner, IterableDataset):
            self.iterable_dataset = inner
        else:
            raise ValueError(
                "Specify either `Dataset` or `IterableDataset` as inner data."
            )

    @property
    def inner(self) -> Dataset | IterableDataset:
        if self.dataset is not None:
            return self.dataset
        elif self.iterable_dataset is not None:
            return self.iterable_dataset
        else:
            assert False, "should not reach here."

    def as_dataset(self) -> Dataset:
        if self.dataset is not None:
            return self.dataset
        elif self.iterable_dataset is not None:
            return _convert_to_non_iterable(self.iterable_dataset)
        else:
            assert False, "should not reach here."

    def as_iterable_dataset(self) -> IterableDataset:
        if self.dataset is not None:
            return self.dataset.to_iterable_dataset()
        elif self.iterable_dataset is not None:
            return self.iterable_dataset
        else:
            assert False, "should not reach here."

    def __len__(self) -> int | None:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return None

    def peek_first(self) -> dict[str, Any]:
        if self.dataset is not None:
            return self.dataset[0]
        elif self.iterable_dataset is not None:
            rows = list(self.iterable_dataset.take(1))
            if len(rows) > 0:
                return rows[0]
            else:
                return None
        else:
            assert False, "should not reach here."

    def filter(self, f, **kwargs):
        if self.dataset is not None:
            return type(self)(self.inner.filter(f, **kwargs))
        elif self.iterable_dataset is not None:
            kwargs.pop("num_proc")
            kwargs.pop("desc")
            return type(self)(self.inner.filter(f, **kwargs))
        else:
            assert False, "should not reach here."

    def map(self, f, **kwargs) -> "DatasetProcessor":
        if self.dataset is not None:
            return type(self)(self.inner.map(f, **kwargs))
        elif self.iterable_dataset is not None:
            kwargs.pop("num_proc")
            kwargs.pop("desc")
            return type(self)(self.inner.map(f, **kwargs))
        else:
            assert False, "should not reach here."

    def remove_columns(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.remove_columns(*args, **kwargs))

    def remove_column(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.remove_column(*args, **kwargs))

    def rename_columns(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.rename_columns(*args, **kwargs))

    def rename_column(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.rename_column(*args, **kwargs))

    def select_columns(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.select_columns(*args, **kwargs))

    def cast_column(self, *args, **kwargs) -> "DatasetProcessor":
        return type(self)(self.inner.cast_column(*args, **kwargs))

    @property
    def column_names(self) -> list[str]:
        if self.dataset is not None:
            return self.dataset.column_names
        elif self.iterable_dataset is not None:
            first = self.peek_first()
            if first is None:
                logger.warn(
                    "tried to look up the columns of an empty iterable dataset."
                )
                return []
            return list(first.keys())
        else:
            assert False, "should not reach here."


def concatenate_dataset_processors(dss: list[DatasetProcessor]) -> DatasetProcessor:
    if len(dss) == 0:
        raise ValueError("must contain at least one dataset")
    if len(dss) == 1:
        return dss[0]
    inner_type: type[Dataset | IterableDataset] = type(dss[0].inner)
    for ds in dss[1:]:
        if not isinstance(ds, inner_type):
            raise ValueError(
                f"all dataset processor to be concatenated must have the same type ({inner_type})"
            )

    return DatasetProcessor(datasets.concatenate_datasets([ds.inner for ds in dss]))
