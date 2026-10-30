# Attribution and provenance

## Upstream foundations named by the owner

- [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime): inference and quantization infrastructure. Upstream repository identifies an MIT license.
- [PyTorch TorchVision](https://github.com/pytorch/vision): vision architectures, pretrained-model interfaces and image transforms. Upstream repository identifies a BSD-3-Clause license.

These projects are libraries/foundations, not a recovered copy of the owner's completed inference service. No source files or weights from them have been imported into this checkout. Exact adopted versions, reused examples and modifications must be recorded from the baseline when it is transferred. Preserve applicable notices for any copied code. Weight and dataset provenance/terms require their own records.

## Existing work and this pass

The owner reports the personal baseline complete on another computer. Source-level ownership, checkpoint lineage, data splits and historical result configuration remain unverified in this checkout. Absence here does not withdraw that completion report.

The new pass proposes an independently written stage-latency investigation and controlled CPU threading experiment, using the existing artifacts and interface. `benchmarks/timing.py` and `benchmarks/report.py` are new independently written measurement helpers in this pass, tested with fabricated records in `tests/fixtures/synthetic/`. They do not contain code imported from the inaccessible personal baseline or either upstream library. Documentation and private lessons are present; baseline/service integration and new measured results remain pending access. Do not present library methods, upstream benchmark figures or toy exercise outputs as personal experimental results.
