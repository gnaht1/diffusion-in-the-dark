import torch

def tail_normalize(image_tensor):
    """
    image_tensor:
        Tensor shape [C,H,W]
        range expected: 0 -> 1
    """

    # avoid sqrt problems
    image_tensor = torch.clamp(image_tensor, min=1e-8)

    # 1. fourth-root transform
    img_fourth_root = torch.pow(image_tensor, 0.25)

    # 2. z-score
    mean = img_fourth_root.mean()
    std = img_fourth_root.std()

    z_scored = (img_fourth_root - mean) / (std + 1e-8)

    # 3. divide by 2
    normalized_img = z_scored / 2.0

    return normalized_img

if __name__ == "__main__":
    x = torch.rand(3, 256, 256)

    y = tail_normalize(x)

    print("Input range:", x.min().item(), x.max().item())
    print("Output range:", y.min().item(), y.max().item())
    print("Mean:", y.mean().item())
    print("Std:", y.std().item())