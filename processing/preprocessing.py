import cv2
import numpy as np
from PIL import Image

def resize_image(image, size=(640, 480)):
    return cv2.resize(image, size)

def normalize_image(image):
    return image / 255.0

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))