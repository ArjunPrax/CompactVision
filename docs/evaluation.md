# Evaluation and reproducibility

Read the [frozen protocol](../IMPROVEMENT_PLAN.md), [configuration](../configs/demo.json) and [record schema](measurement.md). These experiments concern the rebuilt CPU demo, not the previous computer's work.

## Task and separation

A fixed stratified CIFAR-10 subset supplies 1,000 training and 200 validation images per class from official training data, and 200 test images per class from official test data. Calibration samples 128 selected training images. The manifest records source split/index, classes, seed, counts and duplicate skips. Equal integer indices across official train/test identify different images. Selected exact-pixel duplicates are skipped; near-duplicates/upstream pretraining overlap remain limitations.

Only the last MobileNetV3-Small linear layer trains. All other weights remain frozen; deterministic eval-mode features are cached in memory. Highest validation correct count selects an epoch, with validation loss breaking ties. Test labels never select the checkpoint/calibration. Report correct/N, per-class confusions and a paired-image bootstrap interval for INT8-minus-FP32 percentage-point change. Subset results are not comparable to full-dataset published scores as if protocols matched.

## Input, export and quantization

One PIL/NumPy transform in `compact_vision/preprocessing.py` performs RGB, shorter-side 256 bilinear resize, center 224 crop, scaling by 255, prescribed ImageNet means/stds and contiguous float32 NCHW. Decoding materializes pixels/applies EXIF orientation. Tests independently compare the transform with TorchVision across aspect ratios. Every path shares this code.

Parity compares eval-mode PyTorch and ONNX on identical tensors, records numeric errors against frozen tolerances, top-1 disagreements and dynamic batch-two behavior. Structural validity or successful loading is insufficient. INT8 is lossy and receives separate labeled-quality checks.

Static S8S8 QDQ uses per-channel weights and MinMax training-only calibration for supported Conv/Gemm/MatMul nodes, with shape/optimization preparation separate from quantization. Report actual graph/opsets/IR/initializer types and diagnostic optimized kernel/provider events. A quantized filename is not execution evidence. Preserve unsupported/slower/quality-loss findings without forcing a different technology or claiming a speedup.

## Timings

`benchmark` uses a fresh process per condition/repeat, fixed ready validation tensors and explicit warmup. The boundary includes required wrapper conversion/completed output but excludes decode/preprocess/model load/HTTP. Retain each raw observation and repeat.

`study` uses one loopback worker/inference executor and explicit CPU sessions. After development pilots it saves `frozen-plan.json`, then uses identical offered rates/input sequences for the current/candidate pair. An open-arrival scheduler records intended/actual send times, in-flight cap, unsent drops and every terminal outcome. Bounded drain follows the window; the study stops its server afterward. Same-host generation competes for resources and cannot be assumed unlimited.

Server clocks align only within that process: body read, actual executor wait, decode/preprocess/model/postprocess and JSON body construction are measured. Earlier framework/OS/network dispatch, timing-header construction and ASGI/network send are excluded from that server boundary and remain in client end-to-end time. Local gaps are unattributed, not inferred queue time. Diagnostic profiling is separate from final timings.

Raw JSONL may be gzip-compressed; the reporter accepts both. To recompute a real recorded HTTP run, replace `RUN` with its actual directory in the following read-only command (the placeholder is not literal):

```sh
uv run python -m benchmarks.report results/rebuilt-cpu/http/RUN/manifest.json results/rebuilt-cpu/http/RUN/requests.jsonl.gz --traces results/rebuilt-cpu/http/RUN/traces.jsonl.gz
```

## Interpretation limits

Inspect repeat variation, all failure/drop counts, realized offered rate and send lag alongside success p50/p95. Sparse samples/generator caps qualify tails. Thread comparisons use the same FP32 graph; INT8 effects cannot be credited to threading. File bytes are not process RAM. Repeated timings of 32 inputs do not create accuracy examples or production diversity. Native macOS and Docker/Linux are separate contexts.

Keep/change follows the frozen rule, including a negative outcome. This local experiment does not establish real robot safety, workshop accuracy or measured 10×/100× capacity.
