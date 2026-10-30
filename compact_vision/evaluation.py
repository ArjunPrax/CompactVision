"""Task quality on frozen splits; separate INT8 quality from FP32 parity."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .common import environment, object_hash, provenance, read_json, sha256, write_json
from .runtime import make_backend
from .training import Images


def score(config, manifest, split="validation", backends=None):
    if backends is None:
        quantization = Path(config["artifacts"]) / "quantization.json"
        supported = quantization.exists() and read_json(quantization).get("status") == "supported"
        backends = ("torch", "fp32", "int8") if supported else ("torch", "fp32")
    indices = manifest["indices"][split]
    loader = DataLoader(Images(config["data_root"], indices, train=split != "test"), batch_size=config["feature_batch_size"], num_workers=0)
    predictions, targets = {}, []
    for name in backends:
        backend = make_backend(config["artifacts"], name, config["baseline_threads"])
        output = []
        labels = []
        for inputs, batch_labels in loader:
            output.extend(backend.predict(inputs.numpy()).argmax(1).tolist())
            labels.extend(batch_labels.tolist())
        predictions[name], targets = output, labels
        print(f"{split} {name}: {int(np.sum(np.array(output) == targets))}/{len(targets)}", flush=True)
    target = np.asarray(targets)
    report = {"split": split, "n": len(targets), "source_indices": indices, "targets": targets,
              "split_manifest_sha256": object_hash(manifest), "backends": {}, "environment": environment(), "code": provenance()}
    for name, output in predictions.items():
        predicted = np.asarray(output)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (target, predicted), 1)
        report["backends"][name] = {
            "correct": int(np.sum(predicted == target)), "accuracy_pct": float(np.mean(predicted == target) * 100),
            "predictions": output, "confusion_rows_true_columns_predicted": confusion.tolist(),
        }
    if "fp32" in predictions and "int8" in predictions:
        difference = (np.asarray(predictions["int8"]) == target).astype(float) - (np.asarray(predictions["fp32"]) == target)
        rng = np.random.default_rng(config["seed"] + 2)
        bootstrap = np.array([rng.choice(difference, size=len(difference), replace=True).mean() * 100 for _ in range(2000)])
        report["int8_minus_fp32_pp"] = float(difference.mean() * 100)
        report["paired_image_bootstrap_95pct_pp"] = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    if split == "validation":
        loss = -report.get("int8_minus_fp32_pp", float("inf"))
        report["int8_quality_budget_pp"] = config["int8_validation_loss_budget_pp"]
        report["int8_quality_eligible"] = loss <= config["int8_validation_loss_budget_pp"]
    artifacts = Path(config["artifacts"])
    report["artifact_sha256"] = {name: sha256(artifacts / ("checkpoint.pt" if name == "torch" else f"model.{name}.onnx")) for name in backends}
    write_json(Path(config["results"]) / f"quality-{split}.json", report)
    return report
