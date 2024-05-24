import matplotlib.pyplot as plt


def plot_sequences(input, target, prediction):
    """
    Plots the input sequence, target image, and predicted image.

    Args:
        input (numpy.ndarray): The input sequence of images.
        target (numpy.ndarray): The target image.
        prediction (numpy.ndarray): The predicted image.
    """
    plt.figure(figsize=(12, 8))

    # Plot input sequence
    for i in range(input.shape[0]):
        plt.subplot(2, input.shape[0] + 1, i + 1)
        plt.imshow(input[i], cmap='gray')
        plt.title(f'Frame {i + 1}')
        plt.axis('off')

    for i in range(input.shape[0]):
        plt.subplot(2, input.shape[0] + 1, i + input.shape[0] + 2)
        plt.imshow(input[i], cmap='gray')
        plt.title(f'Frame {i + 1}')
        plt.axis('off')

    # Plot target image
    plt.subplot(2, input.shape[0] + 1, input.shape[0] + 1)
    plt.imshow(target, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Plot predicted image
    plt.subplot(2, input.shape[0] + 1, input.shape[0]
                * 2 + 2)  # Adjusted subplot position
    plt.imshow(prediction, cmap='gray')
    plt.title('Predicted Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
