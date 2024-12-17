import torch
import torch.nn as nn

# Define Dice Loss
class DiceLoss(nn.Module):
    def forward(self, outputs, masks, smooth=1e-6):
        outputs = torch.sigmoid(outputs).view(-1)
        masks = masks.view(-1)
        intersection = (outputs * masks).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + masks.sum() + smooth)
        return 1 - dice

dice_loss = DiceLoss()
bce_loss = nn.BCEWithLogitsLoss()

def combined_loss(outputs, masks):
    return 0.5 * dice_loss(outputs, masks) + 0.5 * bce_loss(outputs, masks)
