"""
prepare_dataset.py
-------------------
Quick dataset setup for StyleDiffusion reproduction.

Creates:
data/
 ├── content/
 │     ├── 000.jpg ...
 └── style/
       ├── van_gogh/
       ├── monet/
       └── ukiyoe/
Usage:
    python src/prepare_dataset.py --n_content 100 --size 256
"""

import os, argparse, random
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
from datasets import load_dataset   # huggingface 'datasets' library

# Define styles to download, stored in personal gdrive
STYLE_URLS = {
    "van_gogh": [
        "https://drive.google.com/uc?export=download&id=11jswdbZIc2OutOQ3s71x6xx6PYYDbnKp", # Starry Night
        "https://drive.google.com/uc?export=download&id=1qgxoBVwz79uWD9acVtP8wIk-NBi3xPn3", # Sunflowers
    ],
    "monet": [
        "https://drive.google.com/uc?export=download&id=16tIRnZQT9tZIADCF_NbQpyDuySYKPJUc", # Impression, Sunrise
        "https://drive.google.com/uc?export=download&id=1r5dHwmzTism7fVQA94QGvbqPmpmTXvwI", # Beach at Pourville 
    ],
    "ukiyoe": [
        "https://drive.google.com/uc?export=download&id=1JcnqgKk-L1pFneS2laKHdrAymtHfop5G", # The Great Wave off Kanagawa
        "https://drive.google.com/uc?export=download&id=1FtcFKxst7vkoF_JGrxXpRMpC50XHwhiD", # Red Fuji
    ],
}

def download_style_images(output_dir, size=256):
    for style, urls in STYLE_URLS.items():
        outdir = Path(output_dir) / "style" / style
        outdir.mkdir(parents=True, exist_ok=True)
        for i, url in enumerate(urls):
            r = requests.get(url)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((size, size))
            img.save(outdir / f"{i:03d}.jpg")
    print(f"✅ Saved example styles to {output_dir}/style/")

def download_coco_images(output_dir, n_images=100, size=256):
    """Download subset of COCO training images via HuggingFace."""
    ds = load_dataset("detection-datasets/coco", split="train[:1%]")
    os.makedirs(output_dir, exist_ok=True)
    subset = random.sample(range(len(ds)), min(n_images, len(ds)))
    for i in subset:
        img = Image.open(BytesIO(ds[i]["image"]["bytes"])).convert("RGB")
        img = img.resize((size, size))
        img.save(f"{output_dir}/{i:04d}.jpg")
    print(f"✅ Saved {len(subset)} content images to {output_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Root output directory")
    parser.add_argument("--n_content", type=int, default=100)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    download_style_images(args.out, size=args.size)
    download_coco_images(os.path.join(args.out, "content"),
                         n_images=args.n_content, size=args.size)
    print("🎨 Dataset preparation complete!")

if __name__ == "__main__":
    main()