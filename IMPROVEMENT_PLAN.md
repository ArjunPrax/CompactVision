# CPU inference improvement plan

## Scope and contribution

I designed this study to diagnose latency in a small image-classification service and test one bounded configuration change. Model artifacts, input manifests and run records make the comparison reproducible.

The baseline is ImageNet-pretrained TorchVision MobileNetV3-Small, with only the last linear head trained for CIFAR-10, exported to ONNX and served on CPU. The experiment asks: **where does a batch-one request spend time, and does one alternative thread setting improve it without changing correctness or hiding overload?** A static INT8 experiment evaluates quality, size and backend behavior separately; FP32 remains the default serving backend.

## Milestones and acceptance

| Milestone | Acceptance |
| --- | --- |
| Establish baseline | Fixed manifests; 10,000 training, 2,000 validation and 2,000 official-test images. Train head for 20 epochs; select by validation correct/loss only. |
| Input/export correctness | One transform checked against prescribed upstream transform; FP32 numeric parity on 32 validation images and dynamic batch two. |
| Quantization applicability | Static per-channel S8S8 QDQ, MinMax over 128 training-only images; retain the rejected full-CNN experiment and the one targeted classifier-only correction, with graph/kernel/provider evidence and quality guards. |
| Explain stages | Real decode/preprocess/model/postprocess plus body read, observable executor wait and response-body serialization; exclusions declared. |
| Freeze candidate | Development sweep 1/2/4 CPU threads; choose one nonbaseline alternative on a fixed offered-rate pilot; save rates/rule before final runs. |
| Evaluate | Three fresh-process model repeats; three paired HTTP repeats at three fixed offered rates, with tails, outcome counts, generator validity and variation. |
| Learn independently | Actual source/input trace, small memory reconstruction, prediction-first experiments, one-page explainer and twelve oral questions; learner writes personal answers. |

## Predeclared evaluation contract

- Config/seeds/sizes/quality tolerances are fixed in `configs/demo.json`. Training/calibration/validation/test have separate purposes; calibration is a subset of training. Selected-split exact duplicates are skipped. Near-duplicates and pretrained-data overlap remain limitations.
- FP32 tolerance is `abs(error) <= 1e-4 + 1e-4*abs(reference)`, not argmax-only agreement. INT8 validation-loss budget is two absolute percentage points, fixed before final test scoring. That is a guard, not a predicted result.
- Explicit CPUExecutionProvider; inspect graph IR/opsets, quantized types and diagnostic optimized kernel/provider events. Unsupported/slower INT8 is a valid finding. Dynamic CNN quantization and QAT are not required.
- Matched-budget model context: PyTorch FP32, ONNX FP32, supported ONNX INT8 at batch one/one thread. Thread tuning changes only ORT intra-op threads on the same FP32 graph.
- Model timing: 50 untimed warmups, then 1,000 ready-tensor calls in each of three fresh processes. Retain raw observations; no decode/preprocess/session-load work inside this boundary.
- HTTP inputs are the same ordered 32 validation images encoded as original 32×32 PNGs. One worker and one inference executor remain fixed. Large JPEG/camera decoding and production diversity are not established.
- Pilot each thread setting for eight seconds at 30 req/s after 50 warmups. Choose the alternative with lowest success p95. Baseline serial-active median stage estimates only freeze final rates at 0.5/0.9/1.1 times that estimate; this is not a measured capacity claim.
- Final condition: at least 60 seconds and aim for 2,000 scheduled requests, capped at 180 seconds, followed by bounded client timeout/drain. Three paired repeats, alternating order; max 32 client requests in flight, five-second timeout, retries off. Report realized sample counts.
- Accept the candidate at the lowest-rate nominal point only if median p95 improves at least 5%, its repeat p95 range is below baseline's, success/offered falls by no more than 0.5 pp, neither nominal condition has generator drops, and p95 send lag is below the arrival interval. Otherwise retain baseline. This rule is not a promise of gain.
- Account for every scheduled/sent/drop/success/rejection/error/timeout/outstanding request. Late completion never erases timeout. Separate successful latency, window throughput and cohort drain outcomes. Never add stage p95s or call subtracted percentiles queue time.
- Preserve each repeat, compressed raw records, source/artifact/input/configuration identities and environment. Same-host generation competes for CPU; validity checks qualify conclusions. Three repeats do not establish universal speedups or capacity.

## Boundaries and stopping rule

One model/task/host, head-only CPU training, one static quantization path, one threading decision. No full-backbone training, architecture search, GPU port, serving platform, distributed deployment, dashboard or other projects. The 10×/100× questions remain conceptual.

The study ends once the demo works, correctness and runtime support are checked, and the controlled comparison yields a defensible decision. Final scores must not drive repeated retuning. Docker packaging remains unverified because the evaluation host has no Docker engine. Source and evaluation evidence are public; dataset files and generated weights are excluded from Git.

## Development tolerance diagnosis

The first 32-image FP32 check failed 2/320 elements under absolute tolerance 1e-5, with no top-1 disagreement and maximum absolute error 4.58e-5. A development-only FP64-reference and unoptimized-ORT comparison found ordinary float32 differences of the same order (Torch-to-FP64 max 2.01e-5; optimized ORT-to-FP64 max 4.35e-5; unoptimized ORT-to-Torch max 5.13e-5). The absolute tolerance was revised to 1e-4 on development evidence and frozen before final test scoring; relative tolerance stays 1e-4. Initial failure and diagnostic records are preserved under results/rebuilt-cpu/diagnostics/. An intentionally wrong preprocessing path must still fail the numerical comparison. This change is not a quality-score improvement or a claim of equivalence on all possible inputs.

## Development quantization decision

The original Conv/Gemm/MatMul static path executed integer CPU kernels but scored 249/2000 (12.45%) on validation, versus FP32 1727/2000 (86.35%). Prepared-float output matched original FP32 exactly on the 32-image diagnostic, while both optimized and unoptimized quantized graphs showed the severe loss. This places the problem in the quantized path rather than float graph preparation; it does not establish a specific faulty layer/kernel. One targeted correction leaves the Conv backbone float and quantizes only classifier Gemm operations. It scored 1720/2000 (86.0%), a 0.35 pp validation loss within the unchanged 2 pp guard. The comparison implicates quantization of the convolution path in the larger failure, without resolving per-layer sensitivity. All choices were made on validation data before test scoring. This is partial INT8, not a fully quantized CNN or a guaranteed speedup. Initial records and diagnoses remain under results/rebuilt-cpu/diagnostics/.
