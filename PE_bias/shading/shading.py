import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Function to load MNIST data
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizes pixel values
    ])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return mnist_data

# Function to extract images for a specific digit
def get_digit_images(data, digit, n_images):
    digit_images = [img for img, target in data if target == digit]
    return torch.stack(digit_images[:n_images])

# Function to apply contrast adjustment to a batch of images
def apply_contrast(images, contrast):
    """Applies contrast adjustment to a batch of images, ensuring contrast is between 0 and 1."""
    contrast = max(0, min(contrast, 1))
    images = images.float()  # Assume input is already normalized to [0, 1] or [-1, 1]
    mean_intensity = images.mean(dim=(2, 3), keepdim=True)
    adjusted_images = (images - mean_intensity) * contrast + mean_intensity
    adjusted_images = torch.clamp(adjusted_images, -1, 1)  # Clamping to [-1, 1] to stay within normalized range
    return adjusted_images

# Function to blend two sets of images by taking the maximum value for each pixel
def blend_images_max(image_set1, image_set2):
    return torch.max(image_set1, image_set2)

def create_blended_digit_dataset(data, digit1, digit2, n_images, contrast1, contrast2):
    """
    Creates a new dataset by blending images of two specified digits with individual contrast adjustments.
    Args:
        data: The MNIST dataset loaded with load_mnist_data().
        digit1: The first digit to blend, with a specific contrast adjustment.
        digit2: The second digit to blend, with a different contrast adjustment.
        n_images: Number of images to blend for each digit.
        contrast1: Contrast adjustment for the first digit's images.
        contrast2: Contrast adjustment for the second digit's images.
    
    Returns:
        A tensor dataset of blended images.
    """
    # Extract images for both digits
    images_digit1 = get_digit_images(data, digit1, n_images)
    images_digit2 = get_digit_images(data, digit2, n_images)

    # Apply contrast adjustment to each set of images
    adjusted_images_digit1 = apply_contrast(images_digit1, contrast1)
    adjusted_images_digit2 = apply_contrast(images_digit2, contrast2)

    # Blend images
    blended_images = blend_images_max(adjusted_images_digit1, adjusted_images_digit2)

    return blended_images



