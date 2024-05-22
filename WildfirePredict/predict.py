import torch
import numpy as np

class WildfirePredictor:
    def __init__(self, model, device):
        """
        Initializes the predictor with the model and device.

        Args:
            model: The trained PyTorch model.
            device: The device to run the prediction on (e.g., 'cuda' or 'cpu').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, input_tensor):
        """
        Predicts the output image using the provided model and input tensor.

        Args:
            input_tensor: The input tensor containing the first four images.

        Returns:
            prediction: The predicted output image.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Move input tensor to the correct device
        input_tensor = input_tensor.to(self.device)

        # Predict the fifth image using the first four images
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Convert predicted output to numpy array
        prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
        prediction = (prediction >= 0.5).astype(np.float32)

        return prediction
