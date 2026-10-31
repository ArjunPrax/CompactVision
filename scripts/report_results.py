"""Verify raw run accounting and render the CPU inference study's evidence report.

Run from the repository root: uv run python -m scripts.report_results
This reads recorded evidence; it does not train, generate traffic or retune.
"""

import argparse
from collections import Counter
from pathlib import Path
import statistics

from benchmarks.report import load_json, load_jsonl, summarize
from compact_vision.common import read_json


def median(values):
    return statistics.median(values)


def render(root, *, verify_raw=True):
    root = Path(root)
    quality = read_json(root / "quality-test.json")
    validation = read_json(root / "quality-validation.json")
    initial = read_json(root / "diagnostics/quality-validation-initial-int8.json")
    parity = read_json(root / "parity.json")
    quantization = read_json(root / "quantization.json")
    model = read_json(root / "model/summary.json")
    http = read_json(root / "http/summary.json")
    checked = 0
    if verify_raw:
        for directory in sorted((root / "http").iterdir()):
            if not directory.is_dir():
                continue
            manifest = load_json(directory / "manifest.json")
            recalculated = summarize(manifest, load_jsonl(directory / "requests.jsonl.gz"), load_jsonl(directory / "traces.jsonl.gz"))
            if recalculated != load_json(directory / "summary.json"):
                raise ValueError(f"raw records do not reproduce {directory}/summary.json")
            checked += 1
    plan = http["plan"]
    baseline, candidate = plan["baseline_threads"], plan["candidate_threads"]
    baseline_p95 = http["nominal_p95_ms_by_threads"][str(baseline)]
    candidate_p95 = http["nominal_p95_ms_by_threads"][str(candidate)]
    change = 100 * (1 - median(candidate_p95) / median(baseline_p95))
    fp32_bytes = parity["artifact"]["bytes"]
    int8_bytes = quantization["artifact"]["bytes"]
    sample_run = model["runs"][0]
    env = sample_run["environment"]
    hardware_parts = env["hardware"].split("; ")
    hardware = f"{hardware_parts[0]}, {int(hardware_parts[1]) / 2**30:g} GiB RAM" if len(hardware_parts) == 2 and hardware_parts[1].isdigit() else env["hardware"]
    revisions = sorted({run["summary"]["reproducibility"]["code_revision"] for run in http["runs"]})
    lines = [
        "# CPU classifier: measured results",
        "",
        "Measurements recorded 30 October 2026. I evaluated MobileNetV3-Small on CIFAR-10 and compared CPU inference configurations using the saved protocol and raw run records.",
        "",
        f"**Decision:** {'accept' if http['candidate_accepted'] else 'do not accept'} the {candidate}-thread candidate under the saved rule; recommended setting for this measured host/workload is **{http['recommended_threads']} ORT intra-op threads**. The observed nominal median p95 difference is {change:.1f}% lower for the candidate; the decision also requires repeat-range, success-rate and generator-validity guards. This is not a universal gain.",
        "The thread change uses more CPU parallelism; CPU utilization/energy per request and performance alongside other robotics workloads were not measured.",
        "",
        "## Task quality and artifact size",
        "",
        "Frozen pretrained features; train only the last head on 10,000 images and select epoch/settings on 2,000 validation examples. Final quality uses 2,000 official-test images, 200 per class. Calibration uses 128 selected training inputs. These subset results are not full-CIFAR-10 published benchmarks.",
        "",
        "| Backend | Test correct/N | Test accuracy | Validation correct/N | ONNX artifact bytes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, label in (("torch", "PyTorch FP32"), ("fp32", "ONNX FP32"), ("int8", "Partial INT8: classifier only")):
        row = quality["backends"][name]
        val = validation["backends"].get(name)
        val_text = f"{val['correct']}/{validation['n']}" if val else "see original validation record"
        size = f"{fp32_bytes:,}" if name == "fp32" else f"{int8_bytes:,}" if name == "int8" else "not an ONNX file"
        lines.append(f"| {label} | {row['correct']}/{quality['n']} | {row['accuracy_pct']:.2f}% | {val_text} | {size} |")
    interval = quality["paired_image_bootstrap_95pct_pp"]
    lines.extend([
        "",
        f"Partial INT8 minus FP32 is {quality['int8_minus_fp32_pp']:+.2f} percentage points; paired-image bootstrap 95% interval [{interval[0]:.2f}, {interval[1]:.2f}] pp. It includes zero: the tiny difference is not an established accuracy improvement. The partial graph is {fp32_bytes / int8_bytes:.2f}× smaller in artifact bytes ({100 * (1-int8_bytes/fp32_bytes):.1f}% fewer bytes), **not** fourfold smaller and not a process-RAM measurement. Its convolution backbone remains float.",
        "",
        "### Rejected full-CNN quantization",
        "",
        f"The first Conv/Gemm/MatMul static path scored {initial['backends']['int8']['correct']}/{initial['n']} on validation ({initial['backends']['int8']['accuracy_pct']:.2f}%). It executed CPU integer kernels but failed the unchanged two-percentage-point quality budget. It was rejected before test scoring. Prepared-float outputs matched FP32 on the diagnostic; optimized and unoptimized quantized paths were both poor. One targeted classifier-only alternative recovered validation quality. That implicates convolution-path quantization in the loss, without proving a specific faulty layer/kernel/calibration range.",
        "",
        "### Export correctness",
        "",
        f"Final parity: {parity['n']} class-covering validation images, {parity['element_tolerance_failures']} tolerance failures, {parity['top1_disagreements']} top-1 disagreements; dynamic batch two passed. Absolute/relative tolerances are {parity['atol']}/{parity['rtol']}. The original two near-small-logit failures and FP64/unoptimized-ORT diagnosis are preserved in diagnostics. The tolerance was revised on development evidence before test scoring; an intentionally wrong input transform still fails the regression check. This does not establish equality for every possible input.",
        "",
        "## Ready-tensor model timing",
        "",
        "Each row uses three fresh processes, 50 untimed warmups and 1,000 timed calls per process. p50/p95 below are medians of the three per-run quantiles; brackets show the p95 range. Inputs are the same 32 fixed validation tensors, batch one. This boundary excludes decoding, preprocessing, model/session load and HTTP. Repeated calls do not add accuracy examples.",
        "",
        "| Backend | Intra-op threads | Median p50 ms | Median p95 ms | p95 repeat range ms |",
        "| --- | --- | --- | --- | --- |",
    ])
    conditions = sorted({(run["backend"], run["threads"]) for run in model["runs"]})
    for backend, threads in conditions:
        runs = [r for r in model["runs"] if r["backend"] == backend and r["threads"] == threads]
        p50 = [r["latency_ms"]["p50_ms"] for r in runs]
        p95 = [r["latency_ms"]["p95_ms"] for r in runs]
        lines.append(f"| {backend} | {threads} | {median(p50):.3f} | {median(p95):.3f} | [{min(p95):.3f}, {max(p95):.3f}] |")
    lines.extend([
        "",
        "One-thread rows provide matched-budget backend context; FP32 thread rows are a separate development sweep on the same graph. Eager PyTorch here is not every possible compiled/tuned PyTorch configuration. Partial INT8's storage change does not establish lower request latency.",
        "",
        "## End-to-end loopback HTTP comparison",
        "",
        "One worker, one inference executor, the same FP32 graph and ordered original 32×32 PNG inputs. Candidate/rates/acceptance were frozen after development pilots. Every final condition has a 60-second window, at least 2,000 scheduled arrivals, 50 warmup requests and three paired repeats. Five-second timeout, 32 in-flight client cap and no retries. Successful p50/p95 remain separate from every error/drop outcome. Completions after the window remain cohort success but do not inflate window throughput.",
        "",
        "| Offered req/s | Threads | Median client p50 ms | Median p95 ms [range] | Median window success req/s | Success/offered across repeats | Reject/error/timeout/drop | Max repeat p95 send lag ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for rate_id, rate in enumerate(plan["frozen_offered_rates_rps"]):
        for threads in (baseline, candidate):
            runs = [r["summary"] for r in http["runs"] if r["rate_id"] == rate_id and r["threads"] == threads]
            p50 = [r["client_elapsed_by_outcome_ms"]["success"]["p50_ms"] for r in runs]
            p95 = [r["client_elapsed_by_outcome_ms"]["success"]["p95_ms"] for r in runs]
            counts = {name: sum(r["counts"][name] for r in runs) for name in ("success", "offered", "rejected", "error", "timeout", "generator_drop")}
            outcomes = "/".join(str(counts[n]) for n in ("rejected", "error", "timeout", "generator_drop"))
            lines.append(f"| {rate:.2f} | {threads} | {median(p50):.3f} | {median(p95):.3f} [{min(p95):.3f}, {max(p95):.3f}] | {median(r['successful_completions_per_second_in_window'] for r in runs):.2f} | {counts['success']}/{counts['offered']} | {outcomes} | {max(r['send_lag_ms']['p95_ms'] for r in runs):.3f} |")
    stages_by_threads = {}
    for threads in (baseline, candidate):
        totals, counts = Counter(), Counter()
        for run in [r for r in http["runs"] if r["threads"] == threads and r["rate_id"] == 0]:
            path = root / "http" / run["summary"]["run_id"] / "traces.jsonl.gz"
            for trace in load_jsonl(path):
                per_request = Counter()
                for stage in trace["stages"]:
                    per_request[stage["name"]] += stage["end_ns"] - stage["start_ns"]
                for name, duration in per_request.items():
                    totals[name] += duration
                    counts[name] += 1
        stages_by_threads[threads] = (totals, counts)
    lines.extend(["", "### Nominal server-stage evidence", "",
                  "Mean observed intervals over the nominal traced cohort, not sums of percentile summaries. These are server intervals, not a decomposition of client/network time. Inputs are tiny 32×32 PNGs; larger image decoding can change the bottleneck.", "",
                  f"| Stage | {baseline} thread mean ms (trace count) | {candidate} threads mean ms (trace count) |",
                  "| --- | --- | --- |"])
    for name in ("body_read", "executor_wait", "decode", "preprocess", "model", "postprocess", "serialize"):
        cells = []
        for threads in (baseline, candidate):
            totals, counts = stages_by_threads[threads]
            cells.append(f"{totals[name]/counts[name]/1e6:.4f} ({counts[name]})" if counts[name] else "unobserved")
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines.extend([
        "",
        f"The nominal generator guards {'passed' if http['nominal_generator_validity_passed'] else 'failed'}. The saved decision requires at least 5% median p95 improvement, nonoverlapping repeat p95 ranges, at most 0.5 pp success/offered loss, no nominal client-cap drops and p95 send lag below the arrival interval. Rates were estimated from a serial-stage pilot; they are explored traffic points, not proof of the capacity knee or overload at the highest point.",
        "",
        "### Stage boundaries and instrumentation",
        "",
        "Materialized decoding, preprocessing, model/postprocessing, request-body reading, actual same-process executor wait and JSON body serialization are traced. Earlier framework/OS dispatch, timing-header construction and ASGI/network send are excluded from the server interval and remain in client end-to-end time. No stage p95 addition or cross-machine timestamp subtraction is used. Unattributed local gaps are not automatically queue time.",
        "",
        "The tracing-on/off file contains three short paired diagnostics at 30 req/s. It measures incremental stage/header collection while common wrapper/clock work remains in both modes; short samples and host variability prevent treating that as zero total instrumentation cost. Inspect the raw diagnostic rather than generalizing it to every workload.",
        "",
        "## Reproducibility and limits",
        "",
        f"Recorded host: {hardware}; {env['os']}; architecture {env['architecture']}; Python {env['python']}. Model runs recorded {env['power_source']}. Exact package versions, power snapshots, thread options, source/config/model/input identities and native-host context accompany each run. Frequency/thermal/background load were not pinned. Loopback generation shares CPU with inference; three paired repeats bound the claim rather than establishing universal speedups.",
        f"Measured runtime source revisions: {', '.join(revisions)}. Per-file source digests accompany the records. Later OpenAPI binary-body documentation does not change the measured prediction path.",
        "",
        f"Raw accounting recomputed successfully for {checked} HTTP run directories. Gzip JSONL retains each observation; summaries and final frozen-plan records are versioned. See docs/evaluation.md and docs/measurement.md for reproduction and exact definitions. Generated data/weights remain ignored and local; a clone rebuilds them from the public sources.",
        "",
        "Known limits: tiny CIFAR images/class universe; no high-resolution camera/workshop validation or OOD detector; one seeded frozen-feature head; selected exact-duplicate checks only; unresolved per-layer full-quantization sensitivity; no server-wide admission/deadline mechanism for arbitrary production overload; no measured 10×/100× scaling. Docker packaging is supplied but unverified because this host has no Docker engine. Native results do not validate container performance.",
        "",
        "## Source evidence",
        "",
        "Training/selection: results/rebuilt-cpu/training.json. Input example: input-trace.json. Correctness: parity.json and diagnostics/. Static scope/kernel/weight-parameter evidence: quantization.json. Quality: quality-validation.json and quality-test.json. Model repeats: model/. HTTP pilots, overhead, frozen plan, raw paired runs and decision: http/. These files are relative to the results/rebuilt-cpu root where applicable. Synthetic test fixtures are excluded from these measured claims.",
    ])
    return "\n".join(lines) + "\n", checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/rebuilt-cpu")
    parser.add_argument("--output", default="docs/RESULTS.md")
    args = parser.parse_args()
    body, checked = render(args.root)
    Path(args.output).write_text(body, encoding="utf-8")
    print(f"Verified {checked} raw HTTP cohorts and wrote {args.output}")


if __name__ == "__main__":
    main()
