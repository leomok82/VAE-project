import torch
import numpy as np

def predict_image(model, input_tensor):
    """
    Predicts the output image using the provided model and input tensor.

    Args:
        input_tensor: The input tensor containing the first four images.

    Returns:
        prediction: The predicted output image.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Predict the fifth image using the first four images
    with torch.no_grad():
        prediction = model(input_tensor)

    # Convert predicted output to numpy array
    prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
    prediction = (prediction >= 0.5).astype(np.int16)

    return prediction