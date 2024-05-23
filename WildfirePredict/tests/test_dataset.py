# test_dataset.py
import numpy as np
from WildfirePredict import WildfireDataset

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