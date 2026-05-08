from pathlib import Path
from PIL import Image

# Folder containing this script and source images
folder = Path(__file__).parent

# Supported input image extensions
image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

for img_path in folder.iterdir():
    if img_path.is_file() and img_path.suffix.lower() in image_exts:
        out_path = img_path.with_suffix(".jpg")

        try:
            with Image.open(img_path) as img:
                # Convert to RGB (required for JPEG), resize, and save
                img = img.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                img.save(out_path, "JPEG", quality=95)

            print(f"Converted: {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"Failed: {img_path.name} ({e})")

print("Done.")
