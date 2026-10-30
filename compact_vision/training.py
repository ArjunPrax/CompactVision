"""CPU frozen-feature transfer learning; select the head on validation only."""

import copy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .common import environment, object_hash, provenance, sha256, write_json
from .data import dataset
from .model import WEIGHTS, create_model
from .preprocessing import CONTRACT, preprocess


class Images(Dataset):
    def __init__(self, root, indices, *, train=True):
        self.source = dataset(root, train)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.source[self.indices[position]]
        return torch.from_numpy(preprocess(image)[0]), label


def train_head(config, manifest):
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(config["training_threads"])
    model = create_model(pretrained=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.classifier[-1]
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    model.classifier[-1] = nn.Identity()
    features = {}
    with torch.inference_mode():
        for split in ("train", "validation"):
            loader = DataLoader(Images(config["data_root"], manifest["indices"][split]), batch_size=config["feature_batch_size"], num_workers=0)
            vectors, labels = [], []
            for batch, (inputs, targets) in enumerate(loader):
                vectors.append(model(inputs).clone())
                labels.append(targets)
                if batch % 50 == 0:
                    print(f"Extracting frozen {split} features: batch {batch + 1}/{len(loader)}", flush=True)
            features[split] = (torch.cat(vectors).clone(), torch.cat(labels))
    # Copy out of inference-mode tensors before differentiable linear-head training.
    features = {split: (x.clone(), y.clone()) for split, (x, y) in features.items()}
    loader = DataLoader(TensorDataset(*features["train"]), batch_size=config["head_batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_function = nn.CrossEntropyLoss()
    history, best_state, best_key = [], None, None
    for epoch in range(config["head_epochs"]):
        head.train()
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        head.eval()
        with torch.inference_mode():
            inputs, labels = features["validation"]
            logits = head(inputs)
            correct = int((logits.argmax(1) == labels).sum())
            validation_loss = float(loss_function(logits, labels))
        key = (correct, -validation_loss)
        if best_key is None or key > best_key:
            best_key, best_state = key, copy.deepcopy(head.state_dict())
        row = {"epoch": epoch + 1, "train_loss": total_loss / len(features["train"][1]),
               "validation_correct": correct, "validation_n": len(labels), "validation_loss": validation_loss}
        history.append(row)
        print(f"Head epoch {epoch + 1}: validation {correct}/{len(labels)}", flush=True)
    head.load_state_dict(best_state)
    model.classifier[-1] = head.eval()
    artifacts = Path(config["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "classes": manifest["classes"], "preprocessing": CONTRACT}, checkpoint)
    weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name
    report = {
        "method": "frozen ImageNet pretrained backbone and penultimate classifier; train last linear head only",
        "architecture": "torchvision.mobilenet_v3_small", "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "upstream_weights_url": WEIGHTS.url, "upstream_weights_sha256": sha256(weights_file),
        "checkpoint_sha256": sha256(checkpoint), "split_manifest_sha256": object_hash(manifest),
        "preprocessing": CONTRACT, "config": config, "history": history,
        "selection": "highest validation correct; lowest validation loss breaks ties; test unused",
        "elapsed_seconds": time.perf_counter() - started, "environment": environment(), "code": provenance(),
    }
    write_json(artifacts / "training.json", report)
    return report
