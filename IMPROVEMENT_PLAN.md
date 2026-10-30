# Rebuild and ownership improvement

## Scope and contribution

On 30 October 2026 the owner authorized a full local rebuild because the earlier source cannot be obtained. This supersedes transfer-dependent planning without withdrawing the earlier completion report. This build has independent artifacts, provenance and results.

The baseline is ImageNet-pretrained TorchVision MobileNetV3-Small, with only the last linear head trained for CIFAR-10, exported to ONNX and served on CPU. This public-image demonstration is not workshop-parts recognition. The ownership question is: **where does a batch-one request spend time, and does one alternative thread setting improve it without changing correctness or hiding overload?** A static INT8 experiment supplies separate quality/size/backend context and need not replace FP32.

## Milestones and acceptance

| Milestone | Acceptance |
| --- | --- |
| Rebuild baseline | Fixed manifests; 10,000 training, 2,000 validation and 2,000 official-test images. Train head for 20 epochs; select by validation correct/loss only. |
| Input/export correctness | One transform checked against prescribed upstream transform; FP32 numeric parity on 32 validation images and dynamic batch two. |
| Quantization applicability | Static per-channel S8S8 QDQ, MinMax over 128 training-only images; graph/types/kernel/provider evidence and separate task-quality guard. |
| Explain stages | Real decode/preprocess/model/postprocess plus body read, observable executor wait and response-body serialization; exclusions declared. |
| Freeze candidate | Development sweep 1/2/4 CPU threads; choose one nonbaseline alternative on a fixed offered-rate pilot; save rates/rule before final runs. |
| Evaluate | Three fresh-process model repeats; three paired HTTP repeats at three fixed offered rates, with tails, outcome counts, generator validity and variation. |
| Learn independently | Actual source/input trace, small memory reconstruction, prediction-first experiments, one-page explainer and twelve oral questions; learner writes personal answers. |

## Predeclared evaluation contract

- Config/seeds/sizes/quality tolerances are fixed in `configs/demo.json`. Training/calibration/validation/test have separate purposes; calibration is a subset of training. Selected-split exact duplicates are skipped. Near-duplicates and pretrained-data overlap remain limitations.
- FP32 tolerance is `abs(error) <= 1e-5 + 1e-4*abs(reference)`, not argmax-only agreement. INT8 validation-loss budget is two absolute percentage points, fixed before final test scoring. That is a guard, not a predicted result.
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

Stop once the rebuilt demo works, correctness/support are checked and the bounded comparison yields a defensible decision. Fix demonstrated faults, but do not repeatedly retune on final scores. Report unverified Docker packaging if no Docker engine exists. Keep GitHub private; no data/weight release or website publication.
