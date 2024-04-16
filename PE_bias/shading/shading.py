import torch

from torchvision import datasets, transforms

import numpy as np



def load_mnist_data():

    """Load MNIST dataset with normalization."""

    transform = transforms.Compose([

        transforms.ToTensor(),  # Converts image to tensor.

        transforms.Normalize((0.5,), (0.5,))  # Normalizes pixel values.

    ])

    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    return mnist_data



def get_digit_images(data, digit, n_images):

    """Extracts images for a specific digit."""

    digit_images = [img for img, target in data if target == digit]

    return torch.stack(digit_images[:n_images])




def apply_contrast(images, contrast):
    """Applies contrast adjustment to a batch of images, ensuring contrast is between 0 and 1."""
    # Ensure contrast values are within the bounds [0, 1]
    contrast = max(0, min(contrast, 1))
    
    images = images.float() / 255.0  # Convert to float and normalize to [0, 1]
    mean_intensity = images.mean(dim=(1, 2, 3), keepdim=True)
    adjusted_images = (images - mean_intensity) * contrast + mean_intensity
    adjusted_images = torch.clamp(adjusted_images, 0, 1)  # Ensure values stay within [0, 1]
    return (adjusted_images * 255).byte()  # Convert back to uint8

def blend_images(image_set1, image_set2, proportion1, proportion2):

    return image_set1 * proportion1 + image_set2 * proportion2



def create_blended_dataset(data, digit1, digit2, contrast1, contrast2, proportion1, proportion2, n_images):

    images1 = get_digit_images(data, digit1, n_images)

    images2 = get_digit_images(data, digit2, n_images)

    

    images1_contrasted = apply_contrast(images1, contrast1)

    images2_contrasted = apply_contrast(images2, contrast2)

    

    blended_images = blend_images(images1_contrasted, images2_contrasted, proportion1, proportion2)

    

    return blended_images



# Usage

mnist_data = load_mnist_data()

blended_dataset = create_blended_dataset(mnist_data, digit1=1, digit2=0, contrast1=1.5, contrast2=0.5, proportion1=0.6, proportion2=0.4, n_images=100)






