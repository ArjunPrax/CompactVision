"""Fresh-process model timings and bounded, open-arrival loopback HTTP study."""

import argparse
import asyncio
from contextlib import contextmanager
import io
import gzip
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import httpx
import numpy as np

from benchmarks.report import distribution, summarize
from .common import environment, object_hash, provenance, read_json, write_json
from .data import dataset, development_indices
from .preprocessing import CONTRACT, decode_image, describe, preprocess
from .runtime import Engine, make_backend


def encoded_inputs(config, manifest, count=32):
    source = dataset(config["data_root"], True)
    encoded, identities = [], []
    for index in development_indices(manifest, source, count):
        stream = io.BytesIO()
        source[index][0].save(stream, format="PNG")
        content = stream.getvalue()
        import hashlib
        encoded.append(content)
        identities.append({"official_train_index": index, "true_class_index": source.targets[index], "png_sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "dimensions": [32, 32]})
    return encoded, identities


def save_trace(config, manifest):
    encoded, identities = encoded_inputs(config, manifest, 1)
    image = decode_image(encoded[0])
    tensor = preprocess(image)
    engine = Engine(config["artifacts"])
    prediction, trace, _ = engine.predict_bytes(encoded[0], "documented-input")
    report = {"input": identities[0], "decoded": describe(np.asarray(image)), "tensor": describe(tensor),
              "preprocessing": CONTRACT, "prediction": prediction, "trace": trace,
              "timing_status": "single cold diagnostic, not steady-state benchmark", "code": provenance()}
    write_json(Path(config["results"]) / "input-trace.json", report)
    return report


def model_worker(config_path, backend_name, threads, repeat, output):
    config = read_json(config_path)
    manifest = read_json(Path(config["artifacts"]) / "splits.json")
    encoded, identities = encoded_inputs(config, manifest)
    tensors = [preprocess(decode_image(content)) for content in encoded]
    backend = make_backend(config["artifacts"], backend_name, threads)
    for i in range(config["model_warmup_calls"]):
        backend.predict(tensors[i % len(tensors)])
    rows = []
    for i in range(config["model_timed_calls"]):
        started = time.perf_counter_ns()
        backend.predict(tensors[i % len(tensors)])
        duration = time.perf_counter_ns() - started
        rows.append({"call": i, "input_position": i % len(tensors), "duration_ns": duration})
    output = Path(output)
    with gzip.open(output.with_suffix(".jsonl.gz"), "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    write_json(output, {"evidence_kind": "measured", "backend": backend_name, "threads": threads, "repeat": repeat,
                        "model_sha256": backend.model_sha256, "runtime": backend.info, "batch_size": 1,
                        "warmup_calls": config["model_warmup_calls"], "timed_calls": len(rows),
                        "latency_ms": distribution([row["duration_ns"] for row in rows]),
                        "inputs": identities, "inputs_sha256": object_hash(identities),
                        "int8_scope": read_json(Path(config["artifacts"]) / "quantization.json").get("scope") if backend_name == "int8" else None,
                        "validation_quality_eligible": read_json(Path(config["results"]) / "quality-validation.json").get("int8_quality_eligible") if backend_name == "int8" else True,
                        "boundary": "ready numpy float32 NCHW tensor through completed backend output; includes required wrapper conversion; no decoding/preprocessing/session load",
                        "environment": environment(), "code": provenance()})


def model_study(config_path, config, manifest):
    root = Path(config["results"]) / "model"
    root.mkdir(parents=True, exist_ok=False)
    conditions = [("torch", config["baseline_threads"]), ("fp32", config["baseline_threads"])]
    quantization = Path(config["artifacts"]) / "quantization.json"
    if quantization.exists() and read_json(quantization).get("status") == "supported":
        conditions.append(("int8", config["baseline_threads"]))
    conditions.extend(("fp32", threads) for threads in config["candidate_threads"] if threads != config["baseline_threads"])
    runs = []
    for repeat in range(config["model_repeats"]):
        order = conditions if repeat % 2 == 0 else list(reversed(conditions))
        for backend, threads in order:
            output = root / f"{backend}-t{threads}-r{repeat}.json"
            subprocess.run([sys.executable, "-m", "compact_vision.experiment", "model-worker", str(config_path), backend, str(threads), str(repeat), str(output)], check=True)
            run = read_json(output)
            runs.append(run)
            print(f"Model {backend} t{threads} repeat{repeat}: {run['latency_ms']}", flush=True)
    write_json(root / "summary.json", {"evidence_kind": "measured", "runs": runs,
                                      "interpretation": "matched-budget backend context and separate FP32 thread development sweep; individual repeats retained"})
    return runs


@contextmanager
def running_server(config_path, threads, log_path, *, tracing=True):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(COMPACT_VISION_CONFIG=str(Path(config_path).resolve()), COMPACT_VISION_THREADS=str(threads), COMPACT_VISION_BACKEND="fp32", COMPACT_VISION_TRACING="1" if tracing else "0")
    with Path(log_path).open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "compact_vision.service:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--log-level", "warning", "--no-access-log"], env=env, stdout=log, stderr=log)
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 40
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"service exited; inspect {log_path}")
                try:
                    health = httpx.get(base_url + "/health", timeout=1, trust_env=False)
                    if health.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("service readiness timed out")
                time.sleep(0.05)
            yield base_url, health.json()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


async def traffic(base_url, config, encoded, *, rate, duration, warmup=50, transport=None):
    timeout = config["http_timeout_seconds"]
    limit = config["http_max_in_flight"]
    records, traces, active, all_tasks = [], [], set(), []
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, trust_env=False, transport=transport,
                                limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit)) as client:
        for i in range(warmup):
            response = await client.post("/predict", content=encoded[i % len(encoded)], headers={"content-type": "image/png"})
            response.raise_for_status()
        origin = time.perf_counter_ns()
        duration_ns = int(duration * 1e9)

        async def send(index, scheduled):
            request_id = f"request-{index}"
            sent = time.perf_counter_ns() - origin
            trace = None
            try:
                response = await asyncio.wait_for(client.post("/predict", content=encoded[index % len(encoded)],
                                                             headers={"content-type": "image/png", "x-request-id": request_id}), timeout=timeout)
                finished = time.perf_counter_ns() - origin
                if finished >= sent + int(timeout * 1e9):
                    outcome = "timeout"
                elif response.status_code == 200:
                    payload = response.json()
                    outcome = "success" if isinstance(payload, dict) and payload.get("request_id") == request_id and payload.get("predictions") else "error"
                elif response.status_code in (400, 413, 415, 422, 429, 503):
                    outcome = "rejected"
                else:
                    outcome = "error"
                if "x-compact-trace" in response.headers:
                    trace = json.loads(response.headers["x-compact-trace"])
            except (TimeoutError, httpx.TimeoutException):
                finished = max(time.perf_counter_ns() - origin, sent + int(timeout * 1e9))
                outcome = "timeout"
            except (httpx.HTTPError, ValueError):
                finished = time.perf_counter_ns() - origin
                outcome = "timeout" if finished >= sent + int(timeout * 1e9) else "error"
            records.append({"request_id": request_id, "scheduled_ns": scheduled, "sent_ns": sent,
                            "finished_ns": finished, "outcome": outcome})
            if trace is not None:
                traces.append(trace)

        for index in range(math.ceil(rate * duration)):
            scheduled = int(index / rate * 1e9)
            if scheduled >= duration_ns:
                break
            await asyncio.sleep(max(0, (scheduled - (time.perf_counter_ns() - origin)) / 1e9))
            if len(active) >= limit:
                records.append({"request_id": f"request-{index}", "scheduled_ns": scheduled,
                                "sent_ns": None, "finished_ns": None, "outcome": "generator_drop"})
                continue
            task = asyncio.create_task(send(index, scheduled))
            active.add(task)
            all_tasks.append(task)
            task.add_done_callback(active.discard)
        await asyncio.sleep(max(0, (duration_ns - (time.perf_counter_ns() - origin)) / 1e9))
        if all_tasks:
            await asyncio.gather(*all_tasks)
        observed = time.perf_counter_ns() - origin
    expected = sum(int(index / rate * 1e9) < duration_ns for index in range(math.ceil(rate * duration)))
    return sorted(records, key=lambda row: row["scheduled_ns"]), traces, max(duration_ns, observed), expected


def measured_http_run(config_path, config, manifest, encoded, identities, *, threads, rate, duration, directory, tracing=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with running_server(config_path, threads, directory / "server.log", tracing=tracing) as (url, health):
        records, traces, observed, expected = asyncio.run(traffic(url, config, encoded, rate=rate, duration=duration))
    run_manifest = {
        "schema_version": 1, "run_id": directory.name, "evidence_kind": "measured", "client_clock": "monotonic_offsets_ns",
        "window": {"start_ns": 0, "end_ns": int(duration * 1e9)}, "observation_end_ns": observed,
        "scheduled_requests": expected, "timeout_ns": int(config["http_timeout_seconds"] * 1e9),
        "reproducibility": {
            "model_sha256": health["model_sha256"], "inputs_sha256": object_hash(identities),
            "configuration": {"batch_size": 1, "service_workers": 1, "runtime_threads": {"intra_op": threads, "inter_op": 1},
                              "other_thread_pools": {"inference_executor": 1}, "runtime_options": {**health["runtime"], "trace_collection": tracing}},
            "environment": {**environment(), "runtime": health["runtime"]["version"], "provider": health["runtime"]["providers"],
                            "load_generator": "same-host loopback httpx asyncio; shares CPU budget"},
            "code_revision": provenance()["revision"], "code": provenance(), "warmup_calls": 50,
            "offered_rate_rps": rate, "max_in_flight": config["http_max_in_flight"], "retry_policy": "disabled",
            "timing_boundaries": {"client": "immediately before httpx post to complete response receipt; loopback transport included",
                                  "server": "handler entry through JSON response body construction; excludes framework dispatch before handler, timing-header construction, ASGI send and network",
                                  "executor_wait": "same-process perf_counter_ns from dispatch enqueue to worker entry; explicit observed wait",
                                  "model": "ready numpy float32 tensor through completed ORT CPU output"},
            "input_manifest": identities, "split_manifest_sha256": object_hash(manifest),
        },
    }
    report = summarize(run_manifest, records, traces)
    write_json(directory / "manifest.json", run_manifest)
    for filename, rows in (("requests.jsonl.gz", records), ("traces.jsonl.gz", traces)):
        with gzip.open(directory / filename, "wt") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
    write_json(directory / "summary.json", report)
    print(f"HTTP {directory.name}: {report['client_elapsed_by_outcome_ms']['success']} counts={report['counts']}", flush=True)
    return report


def http_study(config_path, config, manifest):
    root = Path(config["results"]) / "http"
    root.mkdir(parents=True, exist_ok=False)
    encoded, identities = encoded_inputs(config, manifest)
    baseline = config["baseline_threads"]
    overhead = []
    for repeat in range(3):
        for tracing in ([False, True] if repeat % 2 == 0 else [True, False]):
            report = measured_http_run(config_path, config, manifest, encoded, identities, threads=baseline,
                                      rate=30, duration=8, directory=root / f"overhead-{int(tracing)}-r{repeat}", tracing=tracing)
            overhead.append({"repeat": repeat, "trace_collection": tracing, "summary": report})
    write_json(root / "instrumentation.json", {"runs": overhead, "interpretation": "incremental stage collection and trace-header overhead at 30 req/s; common wrapper/clock calls remain in both modes; short diagnostic, not a capacity result"})
    # Development pilots use validation images only; final rates/candidate freeze before final runs.
    pilots = {}
    for threads in [baseline, *config["candidate_threads"]]:
        pilots[threads] = measured_http_run(config_path, config, manifest, encoded, identities, threads=threads,
                                          rate=30, duration=8, directory=root / f"pilot-t{threads}")
    candidate = min(config["candidate_threads"], key=lambda t: pilots[t]["client_elapsed_by_outcome_ms"]["success"]["p95_ms"] or math.inf)
    stages = pilots[baseline]["server_traces"]["by_client_outcome"]["success"]["stages_ms"]
    # With one executor, service capacity is bounded by decode/preprocess/model/postprocess work.
    active_ms = sum(stages[name]["p50_ms"] for name in ("decode", "preprocess", "model", "postprocess"))
    capacity_estimate = 1000 / active_ms
    rates = [round(capacity_estimate * factor, 2) for factor in (0.5, 0.9, 1.1)]
    plan = {"baseline_threads": baseline, "candidate_threads": candidate, "frozen_offered_rates_rps": rates,
            "rate_basis": "development pilot serial-active median stage estimate; not measured sustainable capacity",
            "selection": "lowest success p95 at fixed 30 requests/s development pilot among candidates",
            "acceptance": "nominal (lowest-rate) median p95 improves at least 5%, candidate repeat p95 range lies below baseline range, success/offered does not fall by >0.5pp; neither nominal condition has generator drops or p95 send lag above its arrival interval",
            "quality": "same FP32 graph/input contract; no quality-driven thread tuning", "code": provenance()}
    write_json(root / "frozen-plan.json", plan)
    runs = []
    for repeat in range(config["http_repeats"]):
        for rate_id, rate in enumerate(rates):
            duration = min(config["http_max_duration_seconds"], max(config["http_duration_seconds"], config["http_min_requests"] / rate))
            order = [baseline, candidate] if repeat % 2 == 0 else [candidate, baseline]
            for threads in order:
                report = measured_http_run(config_path, config, manifest, encoded, identities, threads=threads,
                                           rate=rate, duration=duration, directory=root / f"rate{rate_id}-t{threads}-r{repeat}")
                runs.append({"threads": threads, "rate_id": rate_id, "repeat": repeat, "summary": report})
    nominal = {t: [row["summary"] for row in runs if row["threads"] == t and row["rate_id"] == 0] for t in (baseline, candidate)}
    p95 = {t: [r["client_elapsed_by_outcome_ms"]["success"]["p95_ms"] for r in nominal[t]] for t in nominal}
    valid = all(r["counts"]["generator_drop"] == 0 and r["send_lag_ms"]["p95_ms"] < 1000 / rates[0] for rows in nominal.values() for r in rows)
    base_success = min(r["counts"]["success"] / r["counts"]["offered"] for r in nominal[baseline])
    candidate_success = min(r["counts"]["success"] / r["counts"]["offered"] for r in nominal[candidate])
    accepted = (valid and all(value is not None for values in p95.values() for value in values)
                and statistics.median(p95[candidate]) <= 0.95 * statistics.median(p95[baseline])
                and max(p95[candidate]) < min(p95[baseline]) and candidate_success >= base_success - 0.005)
    summary = {"evidence_kind": "measured", "plan": plan, "runs": runs, "nominal_p95_ms_by_threads": p95,
               "nominal_generator_validity_passed": valid, "candidate_accepted": accepted,
               "recommended_threads": candidate if accepted else baseline,
               "interpretation": "retain baseline unless all predeclared nominal guards pass; three same-host repeats do not establish universal capacity or speedup"}
    write_json(root / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["model-worker"])
    parser.add_argument("config")
    parser.add_argument("backend")
    parser.add_argument("threads", type=int)
    parser.add_argument("repeat", type=int)
    parser.add_argument("output")
    arguments = parser.parse_args()
    model_worker(arguments.config, arguments.backend, arguments.threads, arguments.repeat, arguments.output)
