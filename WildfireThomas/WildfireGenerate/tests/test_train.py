import pytest
import numpy as np

from WildfireThomas.WildfireGenerate.task2functions import training

# Train dataset shape: torch.Size([125, 19, 256, 256])
@pytest.fixture
def setup_data():
    data = np.random.rand(100, 256, 256)
    seq_length = 10
    split_size = 5
    batch_size = 16
    device = 'cpu'
    return data, seq_length, split_size, batch_size, device

def test_create_split(setup_data):
    data, seq_length, split_size, _, _ = setup_data
    data_4d = training.create_split(data, seq_length, split_size)
    nb_seq = data.shape[0] // seq_length
    dim = data.shape[1]
    expected_shape = (nb_seq, seq_length // split_size, dim, dim)
    assert data_4d.shape == expected_shape

def test_create_shifted_sequences(setup_data):
    data, seq_length, split_size, _, _ = setup_data
    input_seq, target_seq = training.create_shifted_sequences(data, seq_length, split_size)
    assert input_seq.shape == target_seq.shape
    assert input_seq.shape[1] == seq_length // split_size - 1

def test_WildfiresObjective2(setup_data):
    data, seq_length, split_size, _, _ = setup_data
    dataset = training.WildfiresObjective2(data, seq_length, split_size)
    assert len(dataset) == len(dataset.input_seq)
    assert len(dataset) == len(dataset.target_seq)

def test_create_dataloaders(setup_data):
    data, seq_length, split_size, batch_size, _ = setup_data
    train_loader, train_shifted_loader, _, _ = training.create_dataloaders(data, data, seq_length, split_size, batch_size)
    assert len(train_loader.dataset) == len(data) // (seq_length)
    assert len(train_loader) == len(train_shifted_loader)
