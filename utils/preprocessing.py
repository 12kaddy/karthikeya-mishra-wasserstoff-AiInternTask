from PIL import Image
import cv2
import numpy as np

def resize_image(image_path, target_size=(800, 800)):
    """
    Resizes an image to the specified target size.
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired size as (width, height).
    
    Returns:
        PIL.Image: Resized image.
    """
    image = Image.open(image_path)
    resized_image = image.resize(target_size)
    return resized_image

def convert_to_grayscale(image_path):
    """
    Converts an image to grayscale.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Grayscale image as a NumPy array.
    """
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def normalize_image(image_array):
    """
    Normalizes an image array to have pixel values between 0 and 1.
    
    Args:
        image_array (np.ndarray): Input image as a NumPy array.
    
    Returns:
        np.ndarray: Normalized image array.
    """
    return image_array / 255.0

def preprocess_for_model(image_path):
    """
    Prepares an image for model input by resizing, converting to grayscale, and normalizing.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Preprocessed image ready for model input.
    """
    image = resize_image(image_path)
    grayscale_image = convert_to_grayscale(image_path)
    normalized_image = normalize_image(grayscale_image)
    return normalized_image
