import numpy as np
import torch

import sys
sys.path.append('WildfirPredict')

from WildfireThomas.WildfirePredict.model import ConvLSTMModel, predict_image

def test_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = ConvLSTMModel().to(device)

    # Get the model output
    data = np.random.rand(10, 5, 1, 64, 64).astype(np.float32)
    output = model(torch.tensor(data).to(device))

    # Test output shape
    expected_output_shape = (10, 1, 64, 64)
    assert output.shape == expected_output_shape, f"Output shape {output.shape} is not as expected {expected_output_shape}"

    print(f"Output shape is as expected: {output.shape}")


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