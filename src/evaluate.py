import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils import denormalize

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            if masks is not None:
                masks = masks.to(device)

            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5

            if masks is not None:
                predictions_bool = predictions.bool()
                masks_bool = masks.bool()
                intersection = (predictions_bool & masks_bool).float().sum((1, 2, 3))
                union = (predictions_bool | masks_bool).float().sum((1, 2, 3))
                dice = (2 * intersection) / (predictions_bool.sum((1, 2, 3)) + masks_bool.sum((1, 2, 3)))
                iou = intersection / union

                dice_scores.extend(dice.cpu().numpy())
                iou_scores.extend(iou.cpu().numpy())

            visualize_results(images[0], predictions[0], masks[0] if masks is not None else None)
            break

    if dice_scores:
        print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    if iou_scores:
        print(f"Mean IoU Score: {np.mean(iou_scores):.4f}")

def visualize_results(image, pred_mask, true_mask=None):
    image = denormalize(image.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pred_mask = pred_mask.cpu().squeeze().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="gray")
    plt.title("Predicted Mask")

    if true_mask is not None:
        true_mask = true_mask.cpu().squeeze().numpy()
        plt.subplot(1, 3, 3)
        plt.imshow(true_mask, cmap="gray")
        plt.title("True Mask")
    plt.tight_layout()
    plt.show()
