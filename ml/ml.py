import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


THIS_DIR = Path(__file__).resolve().parent

DEFAULT_OLD_ROOT = THIS_DIR / "numbers-export"
DEFAULT_OLD_LABELS = DEFAULT_OLD_ROOT / "info.labels"
DEFAULT_NEW_ROOTS = (
    THIS_DIR / "Number Hand Gesture Recognition" / "train",
    THIS_DIR / "ASL Digits" / "asl_dataset_digits",
    THIS_DIR / "ASL Digits" / "test",
    THIS_DIR / "more" / "Gesture Image Data",
    THIS_DIR / "more" / "Gesture Image Pre-Processed Data",
)

DEFAULT_TEST_RATIO = 0.2
DEFAULT_BEST_MODEL = THIS_DIR / "best_model.pth"
DEFAULT_LABEL_MAP = THIS_DIR / "label_map.json"
DEFAULT_SPLIT_MANIFEST = THIS_DIR / "merged_split.json"
DEFAULT_ALLOWED_LABELS = ("1", "2", "3", "4", "5")
DEFAULT_MORE_FRACTION = 0.2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sort_label_key(label):
    s = str(label).strip()
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def parse_allowed_labels(labels_csv):
    labels = [s.strip() for s in str(labels_csv).split(",") if s.strip()]
    labels = sorted(set(labels), key=sort_label_key)
    if not labels:
        raise ValueError("allowed labels list is empty")
    return labels


def load_old_json_dataset(old_root, old_labels, allowed_labels):
    samples = []
    with Path(old_labels).open("r") as f:
        rows = json.load(f)["files"]

    old_root = Path(old_root)
    for row in rows:
        label = str(row["label"]["label"]).strip()
        if label not in allowed_labels:
            continue

        rel_path = Path(row["path"])
        image_path = old_root / rel_path

        # Fallback: category + basename if rel path changed.
        if not image_path.exists():
            image_path = old_root / row["category"] / rel_path.name
        if not image_path.exists():
            continue

        samples.append((image_path, label, "old_json"))
    return samples


def root_source_tag(root):
    root = Path(root)
    try:
        rel = root.resolve().relative_to(THIS_DIR.resolve())
        return f"folders:{rel.as_posix()}"
    except ValueError:
        return f"folders:{root.resolve().as_posix()}"


def load_folder_dataset(new_root, allowed_labels, source_tag=None):
    samples = []
    new_root = Path(new_root)
    if not new_root.exists():
        return samples

    if source_tag is None:
        source_tag = root_source_tag(new_root)

    for class_dir in sorted(new_root.iterdir(), key=lambda p: sort_label_key(p.name)):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.strip()
        if label not in allowed_labels:
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, label, source_tag))
    return samples


def merge_and_dedup(samples):
    seen = set()
    merged = []
    for path, label, source in samples:
        key = str(Path(path).resolve())
        if key in seen:
            continue
        seen.add(key)
        merged.append((Path(path), str(label), source))
    return merged


def stratified_split(samples, test_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)

    train, test = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        if n <= 1:
            n_test = 0
        else:
            n_test = int(round(n * test_ratio))
            n_test = max(1, min(n_test, n - 1))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_transforms(train):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.08, 0.08),
                    scale=(0.92, 1.08),
                    shear=6,
                ),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.12,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class SampleDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = [(Path(p), class_to_idx[str(lbl)]) for p, lbl, _ in samples]
        self.transform = transform
        if not self.samples:
            raise RuntimeError("No samples in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx


class TinyGestureNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def epoch_pass(model, loader, criterion, optimizer, device):
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def class_weights_from_samples(samples, class_to_idx, device):
    counts = Counter(class_to_idx[str(lbl)] for _, lbl, _ in samples)
    max_count = max(counts.values())
    weights = torch.ones(len(class_to_idx), dtype=torch.float32)
    for cls_idx in range(len(class_to_idx)):
        cls_count = counts.get(cls_idx, 1)
        weights[cls_idx] = max_count / cls_count
    return weights.to(device)


def summarize_by_label(samples):
    c = Counter(str(lbl) for _, lbl, _ in samples)
    return {k: c[k] for k in sorted(c.keys(), key=sort_label_key)}


def summarize_by_source(samples):
    c = Counter(str(src) for _, _, src in samples)
    return {k: c[k] for k in sorted(c.keys())}


def is_more_root(root):
    root = Path(root).resolve()
    more_root = (THIS_DIR / "more").resolve()
    try:
        root.relative_to(more_root)
        return True
    except ValueError:
        return False


def stable_seed_from_text(text):
    total = 0
    for i, ch in enumerate(str(text)):
        total = (total + (i + 1) * ord(ch)) & 0xFFFFFFFF
    return total


def subsample_by_label(samples, fraction, seed):
    if fraction >= 1.0:
        return list(samples)
    if fraction <= 0.0:
        return []

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[str(sample[1])].append(sample)

    kept = []
    for _, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        keep_n = int(round(len(items) * fraction))
        keep_n = max(1, min(keep_n, len(items)))
        kept.extend(items[:keep_n])

    rng.shuffle(kept)
    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Merge old+new gesture datasets, then split and train."
    )
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--old-labels", type=Path, default=DEFAULT_OLD_LABELS)
    parser.add_argument(
        "--new-roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_NEW_ROOTS),
        help=(
            "One or more folder datasets with class-name subfolders. "
            "All are merged before split/augmentation."
        ),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=None,
        help="Deprecated alias for one extra folder dataset root.",
    )
    parser.add_argument(
        "--allowed-labels",
        type=str,
        default=",".join(DEFAULT_ALLOWED_LABELS),
        help="Comma-separated labels to keep from all sources (default: 1,2,3,4,5).",
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--more-fraction",
        type=float,
        default=DEFAULT_MORE_FRACTION,
        help=(
            "Fraction of samples to keep from roots under ./more "
            "(stratified per label, default: 0.2)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_BEST_MODEL)
    parser.add_argument("--label-map-out", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--split-manifest-out", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()

    if not args.old_labels.exists():
        raise FileNotFoundError(f"Missing old labels file: {args.old_labels}")
    if not args.old_root.exists():
        raise FileNotFoundError(f"Missing old root dir: {args.old_root}")

    new_roots = [Path(p) for p in args.new_roots]
    if args.new_root is not None:
        new_roots.append(Path(args.new_root))

    dedup_new_roots = []
    seen = set()
    for root in new_roots:
        key = str(root.expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup_new_roots.append(root)
    new_roots = dedup_new_roots

    missing_new_roots = [p for p in new_roots if not p.exists()]
    if missing_new_roots:
        missing_text = ", ".join(str(p) for p in missing_new_roots)
        raise FileNotFoundError(f"Missing new root dir(s): {missing_text}")
    if args.more_fraction < 0.0 or args.more_fraction > 1.0:
        raise ValueError("--more-fraction must be in [0.0, 1.0]")

    allowed_labels = set(parse_allowed_labels(args.allowed_labels))

    old_samples = load_old_json_dataset(
        old_root=args.old_root,
        old_labels=args.old_labels,
        allowed_labels=allowed_labels,
    )
    new_samples = []
    new_source_counts_raw = {}
    new_source_counts_used = {}
    for root in new_roots:
        source_tag = root_source_tag(root)
        root_samples = load_folder_dataset(
            new_root=root,
            allowed_labels=allowed_labels,
            source_tag=source_tag,
        )
        new_source_counts_raw[source_tag] = len(root_samples)

        if is_more_root(root):
            local_seed = args.seed + stable_seed_from_text(source_tag)
            root_samples = subsample_by_label(
                root_samples,
                fraction=args.more_fraction,
                seed=local_seed,
            )

        new_source_counts_used[source_tag] = len(root_samples)
        new_samples.extend(root_samples)
    merged = merge_and_dedup(old_samples + new_samples)

    if not merged:
        raise RuntimeError("No merged samples found.")

    train_samples, test_samples = stratified_split(
        merged, test_ratio=args.test_ratio, seed=args.seed
    )
    if not train_samples or not test_samples:
        raise RuntimeError("Split failed; got empty train or test split.")

    labels = sorted({lbl for _, lbl, _ in merged}, key=sort_label_key)
    class_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    idx_to_class = {i: lbl for lbl, i in class_to_idx.items()}

    train_ds = SampleDataset(
        samples=train_samples,
        class_to_idx=class_to_idx,
        transform=build_transforms(train=True),
    )
    test_ds = SampleDataset(
        samples=test_samples,
        class_to_idx=class_to_idx,
        transform=build_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = TinyGestureNet(num_classes=len(class_to_idx)).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Device: {device}")
    print(
        f"Merged samples: {len(merged)} "
        f"(old={len(old_samples)}, new_total={len(new_samples)}, dedup={len(merged)})"
    )
    print(f"more_fraction: {args.more_fraction}")
    print(f"New source counts raw:  {new_source_counts_raw}")
    print(f"New source counts used: {new_source_counts_used}")
    print(f"Merged source counts: {summarize_by_source(merged)}")
    print(f"Train/Test: {len(train_samples)}/{len(test_samples)}")
    print(f"Classes: {class_to_idx}")
    print(f"Label counts train: {summarize_by_label(train_samples)}")
    print(f"Label counts test:  {summarize_by_label(test_samples)}")
    print(f"Parameters: {total_params:,}")

    weights = class_weights_from_samples(train_samples, class_to_idx, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = epoch_pass(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = epoch_pass(model, test_loader, criterion, None, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:5.1f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:5.1f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "test_acc": test_acc,
                },
                args.out,
            )

    args.label_map_out.parent.mkdir(parents=True, exist_ok=True)
    with args.label_map_out.open("w") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            },
            f,
            indent=2,
        )

    args.split_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.split_manifest_out.open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "test_ratio": args.test_ratio,
                "allowed_labels": sorted(allowed_labels, key=sort_label_key),
                "new_roots": [str(p) for p in new_roots],
                "more_fraction": args.more_fraction,
                "new_root_counts_raw": new_source_counts_raw,
                "new_root_counts_used": new_source_counts_used,
                "merged_source_counts": summarize_by_source(merged),
                "train": [
                    {"path": str(p), "label": lbl, "source": src}
                    for p, lbl, src in train_samples
                ],
                "test": [
                    {"path": str(p), "label": lbl, "source": src}
                    for p, lbl, src in test_samples
                ],
            },
            f,
        )

    print(
        f"Best test accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}. "
        f"Saved model: {args.out}"
    )
    print(f"Saved label map: {args.label_map_out}")
    print(f"Saved split manifest: {args.split_manifest_out}")


if __name__ == "__main__":
    main()
