# Current status

Updated 30 October 2026. **The rebuilt native CPU demo and bounded ownership experiment are complete.**

## Scope and artifacts

The owner authorized rebuilding everything because the other computer is unavailable. This is an independently attributed MobileNetV3-Small/CIFAR-10 rebuild, not an exact reproduction of undocumented historical results. The original completion report remains separate. It is a ten-class public-image demo, not workshop-parts recognition or production robot perception.

Source/configs/lock/evaluation evidence are versioned in the private GitHub repository. Public dataset files and generated checkpoints/ONNX weights exist locally under ignored `data/` and `models/demo/`; they are not uploaded. A clone can rebuild them using the documented command. The installed native service can start with `uv run compact-vision serve --threads 4`.

## Verified

- Python 3.12.12, torch 2.15.0, torchvision 0.29.2, ONNX 1.23.2, ORT 1.31.1 locked/installed. CPUExecutionProvider explicitly selected. M4 Pro, 24 GiB RAM, arm64 macOS 26.5.2; recorded runs used battery power.
- Seeded stratified subsets: 10,000 train, 2,000 validation, 2,000 official-test images, 128 training-only calibration inputs; selected-split exact duplicates excluded (zero skips in this selection).
- Frozen backbone/penultimate head features, trained final linear head only. Checkpoint chosen by validation correct/loss; test data did not select training/calibration/quantization settings.
- Shared preprocessing matches the prescribed TorchVision transform on several aspect ratios. Final FP32 parity passed 32 class-covering validation inputs and batch two. Initial two small-logit tolerance failures were diagnosed against FP64/unoptimized ORT; development tolerance revision is preserved/documented before final test scoring.
- Test subset: PyTorch and ONNX FP32 1738/2000 (86.9%); partial classifier-only INT8 1743/2000 (87.15%). Paired difference +0.25 pp, bootstrap interval [-0.05,0.55] pp includes zero; no established accuracy gain.
- Initial full-CNN static quantization executed but failed validation badly (249/2000). One targeted classifier-only correction retained float convolution, scored 1720/2000 on validation and passed the unchanged 2 pp quality guard. The partial artifact is 4,354,835 bytes versus FP32 6,122,049 (28.9% fewer); model-only latency was nearly unchanged. It is not a fully quantized/fourfold-smaller CNN.
- Full native API/checkpoint integration, invalid-image behavior, session reuse, preprocessing, leakage guards and traffic accounting checked: 36 tests pass; Ruff passes. One upstream TestClient deprecation warning remains, without assertion failures.
- Fresh-process model timings: three repeats × 1000 timed calls after 50 warmups for each backend/thread condition. Raw records retained.
- HTTP study: one worker/executor; fixed same-model/input comparison at 69.45, 125.01 and 152.79 req/s, three paired 60-second repeats. 125,016 final offered requests all succeeded, with zero rejection/error/timeout/drop/outstanding outcomes. Window throughput is accounted separately from drain completions.
- Four-thread candidate passed the frozen rule. Nominal p95: baseline repeats [6.236084,6.227125,6.244708] ms; candidate [5.006958,4.972667,4.934833] ms. Median p95 reduction 20.3%, nonoverlapping repeat ranges and generator guards passed. Recommendation is specific to this measured host/workload and uses more CPU parallelism; energy was not measured.
- All 27 raw HTTP cohorts (18 final, 6 tracing diagnostics, 3 pilots) recomputed exactly into their stored summaries. The complete evidence and report-generation command are in docs/RESULTS.md.
- Private beginner curriculum now follows the actual model/service/source and raw records, with separate progressive hints, checks and learner notes. It stays ignored and has a separate local private snapshot; no second-device transfer is implied.

## Limits

No high-resolution workshop/camera, OOD, multiple-training-seed, energy or production-scale validation. Near-duplicate/upstream pretraining overlap and per-layer full-quantization sensitivity remain unresolved. Explored rates did not establish maximum capacity or true overload behavior. Frequency/thermal/background load were not pinned; generation shares host CPU. No server-wide admission/deadline policy for arbitrary production overload. Docker recipe exists but cannot be build/run tested because this host has no Docker engine. No measured container or 10×/100× claim.

The benchmark servers are stopped after the study. Data/weights and learning material remain excluded from Git. Measurement source revision is recorded in each run; a later OpenAPI binary-body declaration changes API documentation only, not the measured prediction path.

## Next action

Use the native demo and work through private lessons 06, 09 and 10. Trace the real input, reconstruct a small mechanism from memory, predict a controlled change, recalculate a metric from raw records, then draw the architecture and answer the twelve questions without notes. Independent understanding is not established by generated guides/tests alone. Future technical work, if requested, should target one remaining evidence gap rather than expanding the platform.
