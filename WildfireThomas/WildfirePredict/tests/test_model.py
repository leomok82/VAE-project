from WildfireThomas.WildfirePredict.model import ConvLSTMModel, predict_image
import numpy as np
import torch

import sys
sys.path.append('WildfirPredict')


def test_model():
    """
    Test the ConvLSTMModel by checking the output shape.

    This function creates an instance of the ConvLSTMModel, passes random data through the model,
    and checks if the output shape matches the expected shape.

    Returns:
        None
    """
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
    """
    Test the predict_image function by checking the output shape and data type.

    This function creates an instance of the ConvLSTMModel, passes random data through the model using the predict_image function,
    and checks if the output shape and data type match the expected values.

    Returns:
        None
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTMModel().to(device)

    # Get the model output
    data = np.random.rand(1, 4, 1, 64, 64).astype(np.float32)
    output = predict_image(model, torch.tensor(data).to(device))

    # Assert the output shape and data type
    assert output.shape == (
        64, 64), f"Output shape is {output.shape}, expected (64, 64)"
    assert output.dtype == np.int16, f"Output data type is {output.dtype}, expected torch.int"

    print(f"Output shape is {output.shape} and data type is {output.dtype}")
