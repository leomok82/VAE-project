import numpy as np
import torch
from WildfirePredict import split_dataset, predict_image, ConvLSTMModel

def test_split_dataset():
    datasize = 10
    data = np.random.rand(datasize, 100, 1, 64, 64) 
    val_ratio = 0.2
    train_dataset, val_dataset = split_dataset(data, val_ratio)

    assert len(train_dataset) == int(datasize * (1 - val_ratio)), \
        f"Training set size is {len(train_dataset)}, expected {int(datasize* (1 - val_ratio))}"
    
    assert len(val_dataset) == int(datasize * val_ratio), \
        f"Validation set size is {len(val_dataset)}, expected {int(datasize * val_ratio)}"
    

def test_predict():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTMModel().to(device)

    # Get the model output
    data = np.random.rand(1, 4, 1, 64, 64).astype(np.float32)
    output = predict_image(model, torch.tensor(data).to(device))

    # Assert the output shape and data type
    assert output.shape == (64, 64), f"Output shape is {output.shape}, expected (64, 64)"
    assert output.dtype == np.int16, f"Output data type is {output.dtype}, expected torch.int"

    print(f"Output shape is {output.shape} and data type is {output.dtype}")