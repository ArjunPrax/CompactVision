# Rebuilt CPU classifier: measured results

30 October 2026. This is the newly rebuilt MobileNetV3-Small/CIFAR-10 experiment, not the inaccessible earlier computer's results. No historical approximate speedup is used as a new measurement.

**Decision:** accept the 4-thread candidate under the saved rule; recommended setting for this measured host/workload is **4 ORT intra-op threads**. The observed nominal median p95 difference is 20.3% lower for the candidate; the decision also requires repeat-range, success-rate and generator-validity guards. This is not a universal gain.
The thread change uses more CPU parallelism; CPU utilization/energy per request and performance alongside other robotics workloads were not measured.

## Task quality and artifact size

Frozen pretrained features; train only the last head on 10,000 images and select epoch/settings on 2,000 validation examples. Final quality uses 2,000 official-test images, 200 per class. Calibration uses 128 selected training inputs. These subset results are not full-CIFAR-10 published benchmarks.

| Backend | Test correct/N | Test accuracy | Validation correct/N | ONNX artifact bytes |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 1738/2000 | 86.90% | see original validation record | not an ONNX file |
| ONNX FP32 | 1738/2000 | 86.90% | 1727/2000 | 6,122,049 |
| Partial INT8: classifier only | 1743/2000 | 87.15% | 1720/2000 | 4,354,835 |

Partial INT8 minus FP32 is +0.25 percentage points; paired-image bootstrap 95% interval [-0.05, 0.55] pp. It includes zero: the tiny difference is not an established accuracy improvement. The partial graph is 1.41× smaller in artifact bytes (28.9% fewer bytes), **not** fourfold smaller and not a process-RAM measurement. Its convolution backbone remains float.

### Rejected full-CNN quantization

The first Conv/Gemm/MatMul static path scored 249/2000 on validation (12.45%). It executed CPU integer kernels but failed the unchanged two-percentage-point quality budget. It was rejected before test scoring. Prepared-float outputs matched FP32 on the diagnostic; optimized and unoptimized quantized paths were both poor. One targeted classifier-only alternative recovered validation quality. That implicates convolution-path quantization in the loss, without proving a specific faulty layer/kernel/calibration range.

### Export correctness

Final parity: 32 class-covering validation images, 0 tolerance failures, 0 top-1 disagreements; dynamic batch two passed. Absolute/relative tolerances are 0.0001/0.0001. The original two near-small-logit failures and FP64/unoptimized-ORT diagnosis are preserved in diagnostics. The tolerance was revised on development evidence before test scoring; an intentionally wrong input transform still fails the regression check. This does not establish equality for every possible input.

## Ready-tensor model timing

Each row uses three fresh processes, 50 untimed warmups and 1,000 timed calls per process. p50/p95 below are medians of the three per-run quantiles; brackets show the p95 range. Inputs are the same 32 fixed validation tensors, batch one. This boundary excludes decoding, preprocessing, model/session load and HTTP. Repeated calls do not add accuracy examples.

| Backend | Intra-op threads | Median p50 ms | Median p95 ms | p95 repeat range ms |
| --- | --- | --- | --- | --- |
| fp32 | 1 | 2.840 | 2.898 | [2.896, 2.935] |
| fp32 | 2 | 1.879 | 1.939 | [1.923, 2.108] |
| fp32 | 4 | 1.542 | 1.599 | [1.586, 1.633] |
| int8 | 1 | 2.869 | 2.909 | [2.898, 2.925] |
| torch | 1 | 8.913 | 9.261 | [9.115, 9.314] |

One-thread rows provide matched-budget backend context; FP32 thread rows are a separate development sweep on the same graph. Eager PyTorch here is not every possible compiled/tuned PyTorch configuration. Partial INT8's storage change does not establish lower request latency.

## End-to-end loopback HTTP comparison

One worker, one inference executor, the same FP32 graph and ordered original 32×32 PNG inputs. Candidate/rates/acceptance were frozen after development pilots. Every final condition has a 60-second window, at least 2,000 scheduled arrivals, 50 warmup requests and three paired repeats. Five-second timeout, 32 in-flight client cap and no retries. Successful p50/p95 remain separate from every error/drop outcome. Completions after the window remain cohort success but do not inflate window throughput.

| Offered req/s | Threads | Median client p50 ms | Median p95 ms [range] | Median window success req/s | Success/offered across repeats | Reject/error/timeout/drop | Max repeat p95 send lag ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 69.45 | 1 | 5.925 | 6.236 [6.227, 6.245] | 69.45 | 12501/12501 | 0/0/0/0 | 1.879 |
| 69.45 | 4 | 4.649 | 4.973 [4.935, 5.007] | 69.45 | 12501/12501 | 0/0/0/0 | 2.144 |
| 125.01 | 1 | 4.915 | 5.125 [5.111, 5.153] | 125.00 | 22503/22503 | 0/0/0/0 | 0.760 |
| 125.01 | 4 | 3.681 | 3.909 [3.909, 3.927] | 125.02 | 22503/22503 | 0/0/0/0 | 0.999 |
| 152.79 | 1 | 4.571 | 4.747 [4.737, 4.769] | 152.78 | 27504/27504 | 0/0/0/0 | 0.544 |
| 152.79 | 4 | 3.461 | 3.663 [3.659, 3.685] | 152.78 | 27504/27504 | 0/0/0/0 | 0.760 |

### Nominal server-stage evidence

Mean observed intervals over the nominal traced cohort, not sums of percentile summaries. These are server intervals, not a decomposition of client/network time. Inputs are tiny 32×32 PNGs; larger image decoding can change the bottleneck.

| Stage | 1 thread mean ms (trace count) | 4 threads mean ms (trace count) |
| --- | --- | --- |
| body_read | 0.0611 (12501) | 0.0679 (12501) |
| executor_wait | 0.0672 (12501) | 0.0757 (12501) |
| decode | 0.0750 (12501) | 0.0935 (12501) |
| preprocess | 0.7491 (12501) | 0.9103 (12501) |
| model | 4.0202 (12501) | 2.3450 (12501) |
| postprocess | 0.0113 (12501) | 0.0150 (12501) |
| serialize | 0.0099 (12501) | 0.0121 (12501) |

The nominal generator guards passed. The saved decision requires at least 5% median p95 improvement, nonoverlapping repeat p95 ranges, at most 0.5 pp success/offered loss, no nominal client-cap drops and p95 send lag below the arrival interval. Rates were estimated from a serial-stage pilot; they are explored traffic points, not proof of the capacity knee or overload at the highest point.

### Stage boundaries and instrumentation

Materialized decoding, preprocessing, model/postprocessing, request-body reading, actual same-process executor wait and JSON body serialization are traced. Earlier framework/OS dispatch, timing-header construction and ASGI/network send are excluded from the server interval and remain in client end-to-end time. No stage p95 addition or cross-machine timestamp subtraction is used. Unattributed local gaps are not automatically queue time.

The tracing-on/off file contains three short paired diagnostics at 30 req/s. It measures incremental stage/header collection while common wrapper/clock work remains in both modes; short samples and host variability prevent treating that as zero total instrumentation cost. Inspect the raw diagnostic rather than generalizing it to every workload.

## Reproducibility and limits

Recorded host: Apple M4 Pro, 24 GiB RAM; macOS-26.5.2-arm64-arm-64bit; architecture arm64; Python 3.12.12. Model runs recorded Now drawing from 'Battery Power'. Exact package versions, power snapshots, thread options, source/config/model/input identities and native-host context accompany each run. Frequency/thermal/background load were not pinned. Loopback generation shares CPU with inference; three paired repeats bound the claim rather than establishing universal speedups.
Measured runtime source revisions: 54e092a35af0e96de4ef714bc5af480a3c100992. Per-file source digests accompany the records. Later OpenAPI binary-body documentation does not change the measured prediction path.

Raw accounting recomputed successfully for 27 HTTP run directories. Gzip JSONL retains each observation; summaries and final frozen-plan records are versioned. See docs/evaluation.md and docs/measurement.md for reproduction and exact definitions. Generated data/weights remain ignored and local; a clone rebuilds them from the public sources.

Known limits: tiny CIFAR images/class universe; no high-resolution camera/workshop validation or OOD detector; one seeded frozen-feature head; selected exact-duplicate checks only; unresolved per-layer full-quantization sensitivity; no server-wide admission/deadline mechanism for arbitrary production overload; no measured 10×/100× scaling. Docker packaging is supplied but unverified because this host has no Docker engine. Native results do not validate container performance.

## Source evidence

Training/selection: results/rebuilt-cpu/training.json. Input example: input-trace.json. Correctness: parity.json and diagnostics/. Static scope/kernel/weight-parameter evidence: quantization.json. Quality: quality-validation.json and quality-test.json. Model repeats: model/. HTTP pilots, overhead, frozen plan, raw paired runs and decision: http/. These files are relative to the results/rebuilt-cpu root where applicable. Historical reports and synthetic test fixtures are excluded from these measured claims.
