import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, List, Any

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        if HAS_TORCH:
            self.normalize_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[Any, np.ndarray]:
        if isinstance(image_input, str):
            cv_img = cv2.imread(image_input)
            if cv_img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                rgb_img = image_input
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
        else:
            raise TypeError("Unsupported image format")

        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        pil_img = Image.fromarray(enhanced_rgb)
        resized_pil = pil_img.resize(self.target_size, Image.BILINEAR)

        if HAS_TORCH:
            tensor = self.normalize_transform(resized_pil).unsqueeze(0)
            return tensor, rgb_img
        else:
            arr = np.array(resized_pil, dtype=np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, 0)
            return arr, rgb_img
