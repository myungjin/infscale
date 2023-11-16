import pytest
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.zoo import Zoo
from tests.module.conftest import model_dataset_pairs


@pytest.mark.parametrize("model_name,dataset_info", model_dataset_pairs)
def test_datasets(model_name, dataset_info):
    mmd = Zoo.get_model_metadata(model_name)
    assert mmd is not None

    dataset_path, dataset_name, split = dataset_info
    dataset = HuggingFaceDataset(mmd, dataset_path, dataset_name, split)
    assert dataset.sample
