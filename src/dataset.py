import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Data augmentation function with CLAHE
def apply_clahe(image):
    image = np.array(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_image)

# Dataset class
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, apply_clahe=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Single-channel mask

        if self.apply_clahe:
            image = apply_clahe(image)

        if self.transform:
            image = self.transform(image)
            mask_transform = Compose([Resize((256, 256)), ToTensor()])
            mask = mask_transform(mask)

        mask = (mask > 0).float()
        return image, mask
