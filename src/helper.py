import numpy as np
import torch
import torchvision.transforms as T

#helper functions
def rgb_to_luma_601(image):
    """Apply ITU-R BT.601-2 luma transform to an RGB image."""
    arr = np.asarray(image).astype(np.float32) / 255.0
    Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return (Y * 255).astype(np.uint8)

def prepare_image_as_tensor(img_pil, image_size=256, device='cuda'):
    # If grayscale (H,W) or single-channel PIL, convert to RGB by repeating channel
    if img_pil.mode == 'L':
        img_pil = img_pil.convert('RGB')
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),                # -> [0,1]
        T.Normalize([0.5]*3, [0.5]*3)  # -> [-1,1]
    ])
    x = transform(img_pil).unsqueeze(0).to(device)  # shape [1,3,H,W]
    return x