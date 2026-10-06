"""
Saree draping classifier (CNN, transfer learning with PyTorch).

Usage:
    python train.py --data_dir data

Expected folder layout:
    data/
        coorg/       *.jpg
        assamese/    *.jpg
        gujarati/    *.jpg
        maharastra/  *.jpg
Class names are taken automatically from the sub-folder names.
"""
import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------ dataset scan
def scan_dataset(data_dir, per_class=200, balance=True, seed=42):
    """Collect valid images, drop corrupt files and exact duplicates, then
    cap every class at `per_class` good images (and optionally trim all
    classes to the size of the smallest one so the dataset is balanced)."""
    root = Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"[ERROR] data_dir '{data_dir}' does not exist.")
    classes = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith("."))
    if len(classes) < 2:
        raise SystemExit("[ERROR] Need at least 2 class sub-folders in data_dir.")

    seen = {}
    good = {c: [] for c in classes}
    stats = {c: {"found": 0, "corrupt": 0, "duplicate": 0, "valid": 0, "kept": 0}
             for c in classes}
    for cname in classes:
        files = sorted(p for p in (root / cname).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMG_EXT)
        for p in files:
            stats[cname]["found"] += 1
            try:
                with Image.open(p) as im:
                    im.load()  # fully decode to detect truncated/corrupt files
            except Exception:
                stats[cname]["corrupt"] += 1
                print(f"  [skip corrupt] {p}")
                continue
            digest = file_md5(p)
            if digest in seen:
                stats[cname]["duplicate"] += 1
                print(f"  [skip duplicate] {p}  (same as {seen[digest]})")
                continue
            seen[digest] = p
            good[cname].append(str(p))
        stats[cname]["valid"] = len(good[cname])

    # --- balance: cap each class at per_class, and (optionally) at the smallest class
    limit = per_class if per_class and per_class > 0 else None
    smallest = min(len(v) for v in good.values())
    if balance:
        limit = smallest if limit is None else min(limit, smallest)
    short = [c for c in classes if per_class and len(good[c]) < per_class]
    if short:
        print("\n[WARNING] Classes with fewer than "
              f"{per_class} valid images: "
              + ", ".join(f"{c} ({len(good[c])})" for c in short))
        if balance:
            print(f"          Balancing -> every class trimmed to {limit} images. "
                  f"Add more images to the short classes to reach {per_class}.")

    rng = random.Random(seed)
    items = []
    for label, cname in enumerate(classes):
        paths = good[cname][:]
        if limit is not None and len(paths) > limit:
            paths = rng.sample(paths, limit)  # reproducible random subset
        stats[cname]["kept"] = len(paths)
        items += [(p, label) for p in sorted(paths)]

    print("\n=== Dataset scan ===")
    print(f"{'class':<14}{'found':>8}{'corrupt':>9}{'duplicate':>11}{'valid':>7}{'kept':>7}")
    for c in classes:
        s_ = stats[c]
        print(f"{c:<14}{s_['found']:>8}{s_['corrupt']:>9}{s_['duplicate']:>11}"
              f"{s_['valid']:>7}{s_['kept']:>7}")
    print(f"Total usable images: {len(items)}")
    return classes, items


def split_dataset(items, val_size, test_size, seed):
    """Stratified split so every class keeps the same proportions."""
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    p_trval, p_test, l_trval, l_test = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    p_train, p_val, l_train, l_val = train_test_split(
        p_trval, l_trval, test_size=rel_val, stratify=l_trval, random_state=seed)
    return (list(zip(p_train, l_train)), list(zip(p_val, l_val)),
            list(zip(p_test, l_test)))


def print_split_table(classes, train, val, test):
    ct, cv, cs = (Counter(l for _, l in s) for s in (train, val, test))
    print("\n=== Images per class in each split ===")
    print(f"{'class':<14}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    for i, c in enumerate(classes):
        print(f"{c:<14}{ct[i]:>8}{cv[i]:>8}{cs[i]:>8}{ct[i]+cv[i]+cs[i]:>8}")
    print(f"{'TOTAL':<14}{len(train):>8}{len(val):>8}{len(test):>8}"
          f"{len(train)+len(val)+len(test):>8}")


def save_split_csv(out_dir, classes, train, val, test):
    for name, split in (("train", train), ("val", val), ("test", test)):
        with open(Path(out_dir) / f"split_{name}.csv", "w", encoding="utf-8") as f:
            f.write("path,label,class\n")
            for p, l in split:
                f.write(f'"{p}",{l},{classes[l]}\n')


class SareeDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # fix phone-photo rotation
        img = img.convert("RGB")            # handles PNG alpha / grayscale / palette
        return self.transform(img), label


def make_transforms(size, hflip):
    train_tf = [transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.75, 1.33))]
    if hflip:
        train_tf.append(transforms.RandomHorizontalFlip())
    train_tf += [
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transforms.Compose(train_tf), eval_tf


# ------------------------------------------------------------------- model
def build_model(arch, n_classes, pretrained):
    if arch in ("resnet18", "resnet50"):
        ctor = getattr(models, arch)
        w = getattr(models, "ResNet18_Weights" if arch == "resnet18"
                    else "ResNet50_Weights").DEFAULT if pretrained else None
        m = ctor(weights=w)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, n_classes))
        head = m.fc
    elif arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(0.4),
                                     nn.Linear(m.classifier[1].in_features, n_classes))
        head = m.classifier
    else:
        raise ValueError(arch)
    return m, head


def set_backbone_trainable(model, head, flag):
    for p in model.parameters():
        p.requires_grad = flag
    for p in head.parameters():
        p.requires_grad = True


# ---------------------------------------------------------- train / evaluate
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, y_true, y_pred, probs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(out.argmax(1).cpu().tolist())
            probs.append(torch.softmax(out.detach(), 1).cpu().numpy())
    n = len(loader.dataset) if not train else len(y_true)
    return (total_loss / n, np.array(y_true), np.array(y_pred),
            np.concatenate(probs))


def train_stage(name, model, epochs, lr, patience, loaders, criterion, device,
                best, history, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bad = 0
    for ep in range(1, epochs + 1):
        tr_loss, yt, yp, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        tr_acc = accuracy_score(yt, yp)
        va_loss, yt, yp, _ = run_epoch(model, loaders["val"], criterion, device)
        va_acc = accuracy_score(yt, yp)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
        flag = ""
        if va_loss < best["loss"]:
            best.update(loss=va_loss, acc=va_acc, state=copy.deepcopy(model.state_dict()))
            bad, flag = 0, "  <-- best"
        else:
            bad += 1
        print(f"[{name}] epoch {ep:02d}/{epochs}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}{flag}")
        if bad >= patience:
            print(f"[{name}] early stopping (no val-loss improvement for {patience} epochs)")
            break


# ------------------------------------------------------------------- plots
def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, probs, classes, path):
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (test set)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(h, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(h["train_loss"], label="train"); axs[0].plot(h["val_loss"], label="val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(h["train_acc"], label="train"); axs[1].plot(h["val_acc"], label="val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet50", "efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--head_epochs", type=int, default=5, help="stage 1: train classifier head only")
    ap.add_argument("--finetune_epochs", type=int, default=30, help="stage 2: fine-tune whole network")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_class", type=int, default=200,
                    help="max good images used per class (0 = no cap)")
    ap.add_argument("--no_balance", action="store_true",
                    help="do NOT trim bigger classes down to the smallest class")
    ap.add_argument("--hflip", action="store_true",
                    help="enable random horizontal flip (OFF by default: mirroring changes the drape direction)")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="train from scratch (not recommended for ~800 images)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # 1) scan + split
    classes, items = scan_dataset(args.data_dir, args.per_class,
                                  not args.no_balance, args.seed)
    train, val, test = split_dataset(items, args.val_size, args.test_size, args.seed)
    print_split_table(classes, train, val, test)
    save_split_csv(out, classes, train, val, test)

    # 2) loaders
    train_tf, eval_tf = make_transforms(args.img_size, args.hflip)
    workers = 0 if os.name == "nt" else 2
    pin = device.type == "cuda"
    loaders = {
        "train": DataLoader(SareeDataset(train, train_tf), batch_size=args.batch_size,
                            shuffle=True, drop_last=True, num_workers=workers, pin_memory=pin),
        "val": DataLoader(SareeDataset(val, eval_tf), batch_size=args.batch_size,
                          shuffle=False, num_workers=workers, pin_memory=pin),
        "test": DataLoader(SareeDataset(test, eval_tf), batch_size=args.batch_size,
                           shuffle=False, num_workers=workers, pin_memory=pin),
    }

    # 3) model + training
    pretrained = not args.no_pretrained
    model, head = build_model(args.arch, len(classes), pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # robust to some noisy labels
    best = {"loss": float("inf"), "acc": 0.0, "state": None}
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n=== Training ({args.arch}, pretrained={pretrained}) ===")
    if pretrained and args.head_epochs > 0:
        set_backbone_trainable(model, head, False)
        train_stage("stage1-head", model, args.head_epochs, args.lr_head, args.patience,
                    loaders, criterion, device, best, history, args.weight_decay)
    set_backbone_trainable(model, head, True)
    lr = args.lr_finetune if pretrained else args.lr_head
    train_stage("stage2-finetune", model, args.finetune_epochs, lr, args.patience,
                loaders, criterion, device, best, history, args.weight_decay)

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "classes": classes, "arch": args.arch,
                "img_size": args.img_size}, out / "best_model.pth")
    plot_history(history, out / "training_curves.png")
    print(f"\nBest val loss {best['loss']:.4f} (val acc {best['acc']:.3f}) -> model saved")

    # 4) final evaluation on TEST set (used exactly once)
    _, y_true, y_pred, probs = run_epoch(model, loaders["test"], criterion, device)
    acc = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    roc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                              labels=list(range(len(classes))))
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    per_class_auc = {c: float(roc_auc_score(y_bin[:, i], probs[:, i]))
                     for i, c in enumerate(classes)}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    report = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)

    print("\n================ TEST SET RESULTS ================")
    print(f"Test images        : {len(y_true)}")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision (macro)  : {prec_m:.4f}   (weighted: {prec_w:.4f})")
    print(f"Recall    (macro)  : {rec_m:.4f}   (weighted: {rec_w:.4f})")
    print(f"F1-score  (macro)  : {f1_m:.4f}   (weighted: {f1_w:.4f})")
    print(f"ROC AUC (OvR macro): {roc_macro:.4f}")
    for c, v in per_class_auc.items():
        print(f"   AUC {c:<12}: {v:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("classes:", classes)
    print(cm)
    print("\nClassification report:")
    print(report)

    plot_confusion(cm, classes, out / "confusion_matrix.png")
    plot_roc(y_true, probs, classes, out / "roc_curves.png")
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc,
                   "precision_macro": prec_m, "recall_macro": rec_m, "f1_macro": f1_m,
                   "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
                   "roc_auc_ovr_macro": roc_macro, "roc_auc_per_class": per_class_auc,
                   "confusion_matrix": cm.tolist(), "classes": classes}, f, indent=2)
    print(f"All outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()