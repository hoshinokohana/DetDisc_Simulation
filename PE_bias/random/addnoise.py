import torch

def add_random_noise(tensor, noise_level):
    """
    Add random Gaussian noise to a single image tensor.

    Parameters:
    - tensor: torch.Tensor, a single image tensor.
    - noise_level: float, the standard deviation of the Gaussian noise.

    Returns:
    - torch.Tensor: the image tensor with added Gaussian noise.
    """
    noise = torch.randn(tensor.size()) * noise_level
    return tensor + noise