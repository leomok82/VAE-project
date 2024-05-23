import numpy as np
import torch
from WildfirePredict import ConvLSTMModel

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
