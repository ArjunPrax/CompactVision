"""Portable identities and explicit run metadata."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    source_files = [*Path("compact_vision").rglob("*.py"), *Path("benchmarks").rglob("*.py"),
                    *Path("configs").glob("*.json"), Path("pyproject.toml"), Path("uv.lock")]
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    known_sources = {str(p) for p in source_files if p.is_file()}
    return {
        "revision": git("rev-parse", "HEAD"),
        "tracked_patch_sha256": hashlib.sha256(git("diff", "HEAD", "--", "compact_vision", "benchmarks", "configs", "pyproject.toml", "uv.lock").encode()).hexdigest(),
        "source_files_sha256": {p: sha256(p) for p in sorted(known_sources)},
        "untracked_source_files": {p: sha256(p) for p in untracked if p in known_sources},
    }


def environment():
    packages = {}
    for name in ("torch", "torchvision", "onnx", "onnxruntime", "numpy", "pillow", "fastapi", "uvicorn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], capture_output=True, text=True) if sys.platform == "darwin" else None
    return {
        "os": platform.platform(), "architecture": platform.machine(), "python": platform.python_version(),
        "hardware": cpu.stdout.strip().replace("\n", "; ") if cpu and cpu.returncode == 0 else platform.processor(),
        "logical_cpus": os.cpu_count(), "packages": packages,
        "execution_context": "native host (no container)",
    }


def load_config(path):
    config = read_json(path)
    for name in ("seed", "train_per_class", "validation_per_class", "test_per_class", "calibration_count",
                 "feature_batch_size", "head_batch_size", "head_epochs", "training_threads", "baseline_threads"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["train_per_class"] + config["validation_per_class"] > 5000 or config["test_per_class"] > 1000:
        raise ValueError("requested per-class split exceeds CIFAR-10 source counts")
    if config["calibration_count"] > 10 * config["train_per_class"]:
        raise ValueError("calibration must fit inside the training subset")
    return config
