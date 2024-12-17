import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Data augmentation function with CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    # Convert PIL Image to NumPy array
    image = np.array(image)
    
    # Convert RGB image to LAB color space for better luminance adjustment
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels (L = lightness, A and B = color channels)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel to enhance brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge the adjusted L channel back with A and B channels
    lab = cv2.merge((l, a, b))
    
    # Convert the LAB image back to RGB color space
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL Image format and return
    return Image.fromarray(enhanced_image)

# Dataset class for handling medical images and masks
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, apply_clahe=False):
        # Directories containing images and masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # List and sort all image and mask files
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        
        # Transformations to apply (e.g., resizing, tensor conversion)
        self.transform = transform
        
        # Flag to apply CLAHE for contrast enhancement
        self.apply_clahe = apply_clahe

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_list)

    def __getitem__(self, idx):
        # Build the file paths for the image and its corresponding mask
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        # Load the image in RGB format and the mask in grayscale (L mode)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply CLAHE to the image for contrast enhancement if the flag is True
        if self.apply_clahe:
            image = apply_clahe(image)

        # Apply transformations to both image and mask
        if self.transform:
            image = self.transform(image)
            
            # Apply specific transformations to the mask (resize and convert to tensor)
            mask_transform = Compose([Resize((256, 256)), ToTensor()])
            mask = mask_transform(mask)

        # Binarize the mask by converting all values > 0 to 1 (foreground)
        mask = (mask > 0).float()
        
        # Return the processed image and its corresponding mask
        return image, mask
