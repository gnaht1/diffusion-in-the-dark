import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from tail_normalization import tail_normalize

INPUT_DIR = "../test_samples"
OUTPUT_DIR = "../degraded_samples"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.ToTensor()


def add_poisson_gaussian_noise(img, poisson_scale=30, gaussian_std=0.02):

    # Poisson noise
    poisson = np.random.poisson(img * poisson_scale) / poisson_scale

    # Gaussian noise
    gaussian = np.random.normal(0, gaussian_std, img.shape)

    noisy = poisson + gaussian

    noisy = np.clip(noisy, 0, 1)

    return noisy


def darken_image(img, brightness_factor=0.2):

    dark = img * brightness_factor
    dark = np.clip(dark, 0, 1)

    return dark


for filename in os.listdir(INPUT_DIR):

    ext = os.path.splitext(filename)[1].lower()
    if ext not in IMAGE_EXTS:
        continue

    path = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(path):
        continue

    pil_img = Image.open(path).convert("RGB")

    img = np.array(pil_img).astype(np.float32) / 255.0

    # darken
    dark_img = darken_image(img)

    # add noise
    noisy_img = add_poisson_gaussian_noise(dark_img)

    # tensor
    tensor_img = transform(noisy_img)

    # tail normalization
    normalized = tail_normalize(tensor_img)

    # save visualization image
    vis = noisy_img * 255
    vis = vis.astype(np.uint8)

    save_name = os.path.splitext(filename)[0] + ".jpg"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # save normalized tensor
    tensor_save_path = os.path.splitext(save_path)[0] + ".pt"
    torch.save(normalized, tensor_save_path)

    print(f"Processed: {filename}")
