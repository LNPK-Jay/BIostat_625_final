
import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from src.dataset import MedicalDataset
from src.model import get_model
from src.loss import combined_loss
from src.train import train_model
from src.evaluate import evaluate_model

# Paths
train_image_dir = "F:/625_final_project/data/training/images"
train_mask_dir = "F:/625_final_project/data/training/1st_manual"
test_image_dir = "F:/625_final_project/data/test_1/images"
test_mask_dir = "F:/625_final_project/data/test_1/1st_manual"

# Transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=mean, std=std)
])

# Data
train_dataset = MedicalDataset(train_image_dir, train_mask_dir, transform, apply_clahe=True)
test_dataset = MedicalDataset(test_image_dir, test_mask_dir, transform, apply_clahe=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model, Optimizer, and Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train and Evaluate
train_model(model, train_loader, combined_loss, optimizer, device, num_epochs=300)
evaluate_model(model, test_loader, device)
