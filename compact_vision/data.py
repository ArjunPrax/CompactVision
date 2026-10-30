"""Deterministic stratified CIFAR-10 subsets, with exact-pixel leakage guards."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

from .common import object_hash, write_json


@lru_cache(maxsize=4)
def dataset(root, train):
    return CIFAR10(root=root, train=train, download=True)


def prepare(config):
    train = dataset(config["data_root"], True)
    test = dataset(config["data_root"], False)
    rng = np.random.default_rng(config["seed"])
    splits = {"train": [], "validation": [], "test": [], "calibration": []}
    seen = set()
    duplicate_skips = 0
    for source, names, counts in (
        (train, ["train", "validation"], [config["train_per_class"], config["validation_per_class"]]),
        (test, ["test"], [config["test_per_class"]]),
    ):
        labels = np.asarray(source.targets)
        for label in range(10):
            candidates = rng.permutation(np.flatnonzero(labels == label)).tolist()
            cursor = 0
            for name, count in zip(names, counts, strict=True):
                selected = 0
                while selected < count:
                    if cursor >= len(candidates):
                        raise ValueError("not enough distinct images for the requested split")
                    index = candidates[cursor]
                    cursor += 1
                    digest = hashlib.sha256(source.data[index].tobytes()).hexdigest()
                    if digest in seen:
                        duplicate_skips += 1
                        continue
                    seen.add(digest)
                    splits[name].append(index)
                    selected += 1
    calibration_rng = np.random.default_rng(config["seed"] + 1)
    for label in range(10):
        candidates = [i for i in splits["train"] if train.targets[i] == label]
        count = config["calibration_count"] // 10 + (label < config["calibration_count"] % 10)
        splits["calibration"].extend(calibration_rng.choice(candidates, size=count, replace=False).tolist())
    for name in splits:
        rng.shuffle(splits[name])
    validate_splits(splits)
    manifest = {
        "dataset": "CIFAR-10 python archive", "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "source_archive_md5": "c58f30108f718f92721af3b95e74349a", "seed": config["seed"],
        "source_split": {"train": "official train", "validation": "official train", "calibration": "official train", "test": "official test"},
        "classes": train.classes, "indices": splits,
        "counts": {name: len(indices) for name, indices in splits.items()},
        "exact_pixel_duplicate_skips": duplicate_skips,
        "limitations": "Subset experiment; no guarantee against near-duplicates or upstream pretraining overlap.",
    }
    write_json(Path(config["artifacts"]) / "splits.json", manifest)
    print(f"Prepared splits {manifest['counts']}; manifest identity {object_hash(manifest)}", flush=True)
    return manifest


def validate_splits(splits):
    for name, indices in splits.items():
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate index inside {name}")
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("training/validation overlap")
    if not set(splits["calibration"]) <= set(splits["train"]):
        raise ValueError("calibration must be a subset of training")
    # Test indices belong to another source split; equal integer indices are not equal images.


def development_indices(manifest, source, count=32):
    """Include each available validation class before filling the fixed corpus."""
    indices = manifest["indices"]["validation"]
    selected, seen_labels = [], set()
    for index in indices:
        label = source.targets[index]
        if label not in seen_labels:
            selected.append(index)
            seen_labels.add(label)
    selected.extend(index for index in indices if index not in selected)
    return selected[:count]
