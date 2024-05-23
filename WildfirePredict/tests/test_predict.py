import numpy as np
import torch
from WildfirePredict import WildfirePredictor, ConvLSTMModel

def test_predict():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    input_dim = 1
    hidden_dim = [64]
    kernel_size = (3, 3)
    num_layers = 1
    output_dim = 1
    model = ConvLSTMModel(input_dim, hidden_dim, kernel_size, num_layers, output_dim).to(device)

    # Get the model output
    data = np.random.rand(10, 5, 1, 64, 64).astype(np.float32)

    predictor = WildfirePredictor(model, device)
    output = predictor.predict_image(torch.tensor(data).to(device))

    # Assert the output shape and data type
    assert output.shape == (10, 1, 64, 64), f"Output shape is {output.shape}, expected (64, 64)"
    assert output.dtype == np.int16, f"Output data type is {output.dtype}, expected torch.int"

    print(f"Output shape is {output.shape} and data type is {output.dtype}")



