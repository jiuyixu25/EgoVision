from torchvision import transforms
from PIL import Image
import cv2

def get_torchvision_augmentation_pipeline():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((480, 640)),  # (height, width)
        transforms.ToTensor(),  # Converts [0,255] PIL Image to [0,1] Tensor
    ])

def apply_augmentation(image_cv2, pipeline=None):
    """
    Apply torchvision-based augmentation pipeline to an OpenCV image.

    Args:
        image_cv2 (np.ndarray): Input image in OpenCV format (BGR).
        pipeline (torchvision.transforms.Compose): Augmentation pipeline.

    Returns:
        torch.Tensor: Augmented image tensor.
    """
    if pipeline is None:
        pipeline = get_torchvision_augmentation_pipeline()

    # Convert BGR OpenCV image to RGB PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

    # Apply augmentation pipeline
    augmented_tensor = pipeline(image_pil)
    return augmented_tensor
