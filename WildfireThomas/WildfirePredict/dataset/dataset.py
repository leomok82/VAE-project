import numpy as np
import torch
from torch.utils.data import Dataset


class WildfireDataset(Dataset):
    def __init__(self, data, window_size=5, step_size=5):
        """
        Initialize the WildfireDataset.

        Args:
            data (numpy.ndarray): Input dataset with shape (num_sequences, seq_length, 1, height, width).
            window_size (int): Length of the window for each input sequence.
            step_size (int): Step size for sliding the window.
        """
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.sequences = []
        self.labels = []

        self._prepare_sequences()

    def _prepare_sequences(self):
        """
        Prepare sequences and labels based on the window size and step size.
        """
        num_sequences, seq_length, channels, height, width = self.data.shape
        interval = 10  # The interval for creating sub-sequences

        for seq_idx in range(num_sequences):
            for offset in range(interval):
                indices = np.arange(offset, seq_length, interval)
                if len(indices) < self.window_size:
                    continue

                for start_idx in range(0, len(indices) - self.window_size + 1, self.step_size):
                    end_idx = start_idx + self.window_size
                    if end_idx > len(indices):
                        break

                    x = self.data[seq_idx, indices[start_idx:end_idx-1]]
                    y = self.data[seq_idx, indices[end_idx-1]]
                    self.sequences.append(x)
                    self.labels.append(y)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Get a specific sequence and its corresponding label.

        Args:
            idx (int): Index of the sequence.

        Returns:
            tuple: A tuple containing the sequence and its corresponding label.
        """
        x = self.sequences[idx]
        y = self.labels[idx]
        return x, y
