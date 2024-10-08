# test_dataset.py
import numpy as np

from WildfireThomas.WildfirePredict.dataset import WildfireDataset, split_dataset


def test_dataset():
    """
    Test the WildfireDataset class.

    This function tests the length and shape of the dataset created using the WildfireDataset class.

    Returns:
        None
    """
    # Instantiate WildfireDataset
    # Assuming you have your data ready
    data = np.random.rand(10, 100, 1, 64, 64)
    window_size = 5
    step_size = 5
    dataset = WildfireDataset(data, window_size, step_size)

    # Test dataset length
    assert len(dataset) == 200, "Dataset length is not as expected"

    # Test shape of the first sequence
    assert dataset[0][0].shape == (
        4, 1, 64, 64), "Shape of the first sequence is not as expected"


def test_split_dataset():
    """
    Test the split_dataset function.

    This function tests the size of the training and validation datasets created using the split_dataset function.

    Returns:
        None
    """
    datasize = 10
    data = np.random.rand(datasize, 100, 1, 64, 64)
    val_ratio = 0.2
    train_dataset, val_dataset = split_dataset(data, val_ratio)

    assert len(train_dataset) == int(datasize * (1 - val_ratio)), \
        f"Training set size is {len(train_dataset)}, expected {int(datasize* (1 - val_ratio))}"

    assert len(val_dataset) == int(datasize * val_ratio), \
        f"Validation set size is {len(val_dataset)}, expected {int(datasize * val_ratio)}"
