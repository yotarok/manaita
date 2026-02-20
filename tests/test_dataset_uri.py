import unittest.mock

from manaita import dataset_uri


def test_dataset_uri(fake_dataset):
    mock_load_dataset = unittest.mock.MagicMock(return_value=fake_dataset)
    dataset_uri.register_data_source(
        "mock", override_default=True, allow_override=True
    )(mock_load_dataset)
    dsid = dataset_uri.DatasetUri.from_string("karpathy/tiny_shakespeare")
    dsid.load_dataset()

    assert len(mock_load_dataset.call_args_list) == 1
    uri = mock_load_dataset.call_args_list[-1].args[0]

    assert uri.name == "karpathy/tiny_shakespeare"
    assert uri.parameters == []
    assert uri.split is None

    # Again, with full specification.
    dsid = dataset_uri.DatasetUri.from_string("mock:karpathy/tiny_shakespeare:train")
    dsid.load_dataset()

    assert len(mock_load_dataset.call_args_list) == 2
    uri = mock_load_dataset.call_args_list[-1].args[0]

    assert uri.name == "karpathy/tiny_shakespeare"
    assert uri.parameters == []
    assert uri.split == "train"


def test_dataset_uri_split(fake_dataset):
    mock_load_dataset = unittest.mock.MagicMock(return_value=fake_dataset)
    dataset_uri.register_data_source(
        "mock", override_default=True, allow_override=True
    )(mock_load_dataset)
    dsid = dataset_uri.DatasetUri.from_string("karpathy/tiny_shakespeare:train")
    dsid.load_dataset()

    # TODO: currently "train[0:20]" split is not supported when scheme is not specified.

    assert len(mock_load_dataset.call_args_list) == 1

    uri = mock_load_dataset.call_args_list[-1].args[0]
    assert uri.name == "karpathy/tiny_shakespeare"
    assert uri.parameters == []
    assert uri.split == "train"


def test_dataset_uri_data_dir(fake_dataset):
    mock_load_dataset = unittest.mock.MagicMock(return_value=fake_dataset)
    dataset_uri.register_data_source(
        "mock", override_default=True, allow_override=True
    )(mock_load_dataset)
    dsid = dataset_uri.DatasetUri.from_string(
        "karpathy/tiny_shakespeare:train?data_dir=/home/dataset"
    )
    dsid.load_dataset()

    # TODO: currently "train[0:20]" split is not supported when scheme is not specified.

    assert len(mock_load_dataset.call_args_list) == 1

    uri = mock_load_dataset.call_args_list[-1].args[0]
    assert uri.name == "karpathy/tiny_shakespeare"
    assert uri.parameters == [("data_dir", "/home/dataset")]
    assert uri.split == "train"
