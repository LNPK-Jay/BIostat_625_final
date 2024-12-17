import numpy as np

def denormalize(image, mean, std):
    image = image.numpy().transpose(1, 2, 0)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image
