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
    
    # Convert RGB image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (brightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge modified L channel back with A and B channels
    lab = cv2.merge((l, a, b))
    
    # Convert LAB image back to RGB color space
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Convert back to PIL Image and return
    return Image.fromarray(enhanced_image)

# Dataset class for medical images and masks
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, apply_clahe=False):
        # Initialize dataset with directories, transformations, and CLAHE flag
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))  # List and sort image files
        self.mask_list = sorted(os.listdir(mask_dir))    # List and sort mask files
        self.transform = transform                      # Transformations for images
        self.apply_clahe = apply_clahe                  # Flag to apply CLAHE

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_list)

    def __getitem__(self, idx):
        # Get the file paths for the image and corresponding mask
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        # Load the image and mask using PIL
        image = Image.open(image_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")      # Convert mask to single-channel grayscale

        # Apply CLAHE to the image if the flag is set
        if self.apply_clahe:
            image = apply_clahe(image)

        # Apply transformations to the image and mask
        if self.transform:
            image = self.transform(image)
            
            # Define and apply specific transformations for the mask
            mask_transform = Compose([Resize((256, 256)), ToTensor()])
            mask = mask_transform(mask)

        # Binarize the mask (convert to 0 or 1)
        mask = (mask > 0).float()
        
        # Return the processed image and mask as a tuple
        return image, mask
