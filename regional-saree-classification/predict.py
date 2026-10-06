"""
predict.py - classify new saree images with the trained model and SHOW them.

Usage (from the saree_project folder):
    python predict.py                          # uses test_images/ , one image per window
    python predict.py --mode grid              # all images in one window
    python predict.py --images_dir my_folder   # different folder
    python predict.py --save                   # also save an overview picture

Put the new images in:  test_images/   (any of .jpg .jpeg .jfif .png .webp .bmp .tif)
Nothing is written to CSV - the images themselves are displayed with the prediction on top.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch, n_classes):
    """Same architecture as train.py (weights come from the checkpoint)."""
    if arch in ("resnet18", "resnet50"):
        m = getattr(models, arch)(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, n_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4),
                                     nn.Linear(m.classifier[1].in_features, n_classes))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, ckpt["img_size"]


def predict_image(model, tf, path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0].numpy()
    return img, probs


def show_single(img, name, classes, probs, idx, total):
    """Big image on the left, probability bars on the right."""
    best = int(probs.argmax())
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"Predicted: {classes[best].upper()}  ({probs[best]*100:.1f}%)",
                     fontsize=15, fontweight="bold",
                     color="green" if probs[best] >= 0.6 else "darkorange")
    colors = ["tab:green" if i == best else "lightgray" for i in range(len(classes))]
    ax_bar.barh(classes, probs * 100, color=colors)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.invert_yaxis()
    for i, p in enumerate(probs):
        ax_bar.text(min(p * 100 + 1, 82), i, f"{p*100:.1f}%", va="center")
    fig.suptitle(f"[{idx}/{total}]  {name}", fontsize=10)
    fig.tight_layout()
    return fig


def show_grid(results, classes, cols=4):
    n = len(results)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (name, img, probs) in zip(axs.ravel(), results):
        best = int(probs.argmax())
        ax.imshow(img)
        ax.set_title(f"{classes[best]} ({probs[best]*100:.0f}%)\n{name[:28]}", fontsize=10,
                     color="green" if probs[best] >= 0.6 else "darkorange")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="test_images")
    ap.add_argument("--model", default="results/best_model.pth")
    ap.add_argument("--mode", choices=["single", "grid"], default="single",
                    help="single = one big window per image (close it to see next); grid = all together")
    ap.add_argument("--per_page", type=int, default=12,
                    help="grid mode: images per window (default 12 = 4 x 3)")
    ap.add_argument("--save", action="store_true",
                    help="also save one overview picture (results/predictions.png). Off by default.")
    ap.add_argument("--out", default="results/predictions.png")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        raise SystemExit(f"[ERROR] Folder '{img_dir}' not found. Create it and put images inside.")
    if not Path(args.model).is_file():
        raise SystemExit(f"[ERROR] Model '{args.model}' not found. Run train.py first.")
    files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"[ERROR] No images found in '{img_dir}'.")

    model, classes, size = load_model(args.model)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    print(f"Model loaded. Classes: {classes}\nTesting {len(files)} image(s) from '{img_dir}'\n")

    results = []
    for i, f in enumerate(files, 1):
        try:
            img, probs = predict_image(model, tf, f)
        except Exception as e:
            print(f"[skip unreadable] {f.name}: {e}")
            continue
        best = int(probs.argmax())
        print(f"{f.name:<45} -> {classes[best]:<12} {probs[best]*100:5.1f}%")
        results.append((f.name, img, probs))
        if args.mode == "single":
            fig = show_single(img, f.name, classes, probs, i, len(files))
            plt.show()          # blocks until the window is closed -> then next image
            plt.close(fig)

    if not results:
        raise SystemExit("[ERROR] No readable images.")

    if args.mode == "grid" or args.save:
        pages = [results[i:i + args.per_page] for i in range(0, len(results), args.per_page)]
        for n, chunk in enumerate(pages, 1):
            grid = show_grid(chunk, classes)
            grid.suptitle(f"Page {n}/{len(pages)}", fontsize=14)
            if args.save:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                p = Path(args.out).with_name(f"predictions_page{n}.png")
                grid.savefig(p, dpi=120)
                print(f"Saved: {p.resolve()}")
            if args.mode == "grid":
                plt.show()      # close the window to go to the next page
            plt.close(grid)


if __name__ == "__main__":
    main()