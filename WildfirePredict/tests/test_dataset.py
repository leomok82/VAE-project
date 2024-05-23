# test_dataset.py
import numpy as np
from WildfirePredict.dataset import WildfireDataset, split_dataset

def test_dataset():
    # Instantiate WildfireDataset
    data = np.random.rand(10, 100, 1, 64, 64)  # Assuming you have your data ready
    window_size = 5
    step_size = 5
    dataset = WildfireDataset(data, window_size, step_size)

    # Test dataset length
    assert len(dataset) == 200, "Dataset length is not as expected"

    # Test shape of the first sequence
    assert dataset[0][0].shape == (4, 1, 64, 64), "Shape of the first sequence is not as expected"

def test_split_dataset():
    datasize = 10
    data = np.random.rand(datasize, 100, 1, 64, 64) 
    val_ratio = 0.2
    train_dataset, val_dataset = split_dataset(data, val_ratio)

    assert len(train_dataset) == int(datasize * (1 - val_ratio)), \
        f"Training set size is {len(train_dataset)}, expected {int(datasize* (1 - val_ratio))}"
    
    assert len(val_dataset) == int(datasize * val_ratio), \
        f"Validation set size is {len(val_dataset)}, expected {int(datasize * val_ratio)}"