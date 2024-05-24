import torch
from torch.utils.data import random_split

def split_dataset(dataset, val_ratio=0.2, seed=None):
    """
    Splits a dataset into training and validation sets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        val_ratio (float): The ratio of the dataset to use for validation.
        seed (int, optional): A random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training dataset and the validation dataset.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

# Example usage:
# train_dataset, val_dataset = split_dataset(train_dataset, val_ratio=0.2, seed=42)
