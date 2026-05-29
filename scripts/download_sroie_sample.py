from pathlib import Path
from datasets import load_dataset

out = Path("data/sroie_sample")
out.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("jsdnrs/ICDAR2019-SROIE", split="test")

limit = 25
for i, row in enumerate(dataset.select(range(limit))):
    image = row["image"]
    image_path = out / f"sroie_{i:04d}.jpg"
    image.save(image_path)

    meta = {k: v for k, v in row.items() if k != "image"}
    meta_path = out / f"sroie_{i:04d}.txt"
    meta_path.write_text(str(meta), encoding="utf-8")

print(f"Saved {limit} SROIE images to {out}")
