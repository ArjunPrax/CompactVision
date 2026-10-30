"""Validate one run's client records and summarize outcomes without dropping failures.

Run with ``python3 -m benchmarks.report MANIFEST REQUESTS [--traces TRACES]``.
This module does not generate traffic, load a model or claim export/task correctness.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from .timing import identifier, nanoseconds, nonnegative_integer, validate_trace

OUTCOMES = ("success", "rejected", "error", "timeout", "outstanding", "generator_drop")
METADATA_FIELDS = (
    "model_sha256", "inputs_sha256", "configuration", "environment", "code_revision",
    "warmup_calls", "offered_rate_rps", "max_in_flight", "retry_policy", "timing_boundaries",
)
NESTED_METADATA = {
    "configuration": ("batch_size", "service_workers", "runtime_threads", "other_thread_pools", "runtime_options"),
    "environment": ("hardware", "os", "architecture", "python", "runtime", "provider"),
    "timing_boundaries": ("client", "server"),
}


def distribution(values_ns):
    """Median p50 and nearest-rank p95, in ms; no p99 or silent empty zeros."""
    values = sorted(values_ns)
    return {
        "count": len(values),
        "p50_ms": statistics.median(values) / 1_000_000 if values else None,
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1] / 1_000_000 if values else None,
    }


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("manifest.schema_version must be 1")
    identifier(manifest.get("run_id"), "manifest.run_id")
    if manifest.get("evidence_kind") not in ("synthetic", "measured"):
        raise ValueError("evidence_kind must explicitly be synthetic or measured")
    if manifest.get("client_clock") != "monotonic_offsets_ns":
        raise ValueError("client_clock must be monotonic_offsets_ns")
    window = manifest.get("window")
    if not isinstance(window, dict):
        raise ValueError("manifest.window must be an object")
    start = nanoseconds(window.get("start_ns"), "window.start_ns")
    end = nanoseconds(window.get("end_ns"), "window.end_ns")
    observed = nanoseconds(manifest.get("observation_end_ns"), "observation_end_ns")
    if not start < end <= observed:
        raise ValueError("window must have positive duration and end before observation ends")
    nonnegative_integer(manifest.get("scheduled_requests"), "scheduled_requests")
    if nanoseconds(manifest.get("timeout_ns"), "timeout_ns") == 0:
        raise ValueError("timeout_ns must be positive")
    metadata = manifest.get("reproducibility", {})
    if not isinstance(metadata, dict):
        raise ValueError("reproducibility must be an object")
    missing = [key for key in METADATA_FIELDS if key not in metadata or metadata[key] is None]
    for key in ("model_sha256", "inputs_sha256"):
        if key in metadata and metadata[key] is not None:
            value = metadata[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"reproducibility.{key} must be a lowercase SHA-256 hex digest")
    for key in ("configuration", "environment", "timing_boundaries"):
        if key in metadata and (not isinstance(metadata[key], dict) or not metadata[key]):
            raise ValueError(f"reproducibility.{key} must be a nonempty object")
        if key in metadata:
            for field in NESTED_METADATA[key]:
                if field not in metadata[key] or metadata[key][field] is None or metadata[key][field] == "":
                    missing.append(f"{key}.{field}")
    configuration = metadata.get("configuration", {})
    for key in ("batch_size", "service_workers"):
        if key in configuration and nonnegative_integer(configuration[key], key) == 0:
            raise ValueError(f"configuration.{key} must be positive")
    for key in ("runtime_threads", "other_thread_pools", "runtime_options"):
        if key in configuration and not isinstance(configuration[key], dict):
            raise ValueError(f"configuration.{key} must be an object (empty if explicitly inapplicable)")
    if "code_revision" in metadata:
        identifier(metadata["code_revision"], "code_revision")
    if "warmup_calls" in metadata:
        nonnegative_integer(metadata["warmup_calls"], "warmup_calls")
    if "max_in_flight" in metadata:
        if nonnegative_integer(metadata["max_in_flight"], "max_in_flight") == 0:
            raise ValueError("max_in_flight must be positive")
    if "offered_rate_rps" in metadata:
        rate = metadata["offered_rate_rps"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("offered_rate_rps must be finite and positive")
    if "retry_policy" in metadata and metadata["retry_policy"] != "disabled":
        raise ValueError("schema 1 supports retry_policy=disabled only; one row per scheduled request")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("reproducibility must contain finite JSON values") from error
    if manifest["evidence_kind"] == "measured" and missing:
        raise ValueError("measured run missing reproducibility fields: " + ", ".join(missing))
    return missing


def validate_request(row, manifest):
    request_id = identifier(row.get("request_id"), "request_id")
    scheduled = nanoseconds(row.get("scheduled_ns"), f"{request_id}.scheduled_ns")
    window = manifest["window"]
    if not window["start_ns"] <= scheduled < window["end_ns"]:
        raise ValueError(f"{request_id}: scheduled arrival must be in [window start, end)")
    outcome = row.get("outcome")
    if outcome not in OUTCOMES:
        raise ValueError(f"{request_id}: unknown outcome")
    sent, finished = row.get("sent_ns"), row.get("finished_ns")
    if outcome == "generator_drop":
        if sent is not None or finished is not None:
            raise ValueError(f"{request_id}: generator_drop must be unsent and unfinished")
        return
    sent = nanoseconds(sent, f"{request_id}.sent_ns")
    observed = manifest["observation_end_ns"]
    if not scheduled <= sent <= observed:
        raise ValueError(f"{request_id}: send must follow schedule and precede observation end")
    deadline = sent + manifest["timeout_ns"]
    if outcome == "outstanding":
        if finished is not None or observed >= deadline:
            raise ValueError(f"{request_id}: outstanding must be unfinished and not yet timed out")
        return
    finished = nanoseconds(finished, f"{request_id}.finished_ns")
    if not sent <= finished <= observed:
        raise ValueError(f"{request_id}: finish must follow send and precede observation end")
    if outcome == "timeout":
        if finished < deadline:
            raise ValueError(f"{request_id}: timeout cannot be reported before its deadline")
    elif finished >= deadline:
        raise ValueError(f"{request_id}: response at/after client deadline must remain a timeout")


def summarize(manifest, requests, traces=()):
    """One run/repeat only. Never pool configurations or infer a speedup."""
    missing = validate_manifest(manifest)
    if len(requests) != manifest["scheduled_requests"]:
        raise ValueError("request row count must equal scheduled_requests; missing arrivals cannot disappear")
    by_id = {}
    counts = Counter({outcome: 0 for outcome in OUTCOMES})
    elapsed = defaultdict(list)
    send_lag = []
    window = manifest["window"]
    in_window_success = 0
    for row in requests:
        if not isinstance(row, dict):
            raise ValueError("request rows must be objects")
        validate_request(row, manifest)
        request_id = row["request_id"]
        if request_id in by_id:
            raise ValueError(f"duplicate request_id: {request_id}")
        by_id[request_id] = row
        counts[row["outcome"]] += 1
        if row["outcome"] != "generator_drop":
            send_lag.append(row["sent_ns"] - row["scheduled_ns"])
        if row["outcome"] not in ("outstanding", "generator_drop"):
            elapsed[row["outcome"]].append(row["finished_ns"] - row["sent_ns"])
        if row["outcome"] == "success" and window["start_ns"] <= row["finished_ns"] < window["end_ns"]:
            in_window_success += 1

    server = defaultdict(lambda: {"duration": [], "unattributed": [], "stages": defaultdict(list)})
    seen_traces = set()
    for trace in traces:
        if not isinstance(trace, dict):
            raise ValueError("trace rows must be objects")
        validate_trace(trace)
        request_id = trace["request_id"]
        if request_id not in by_id or request_id in seen_traces:
            raise ValueError("each trace must identify a unique scheduled request")
        if by_id[request_id]["outcome"] == "generator_drop":
            raise ValueError("an unsent generator_drop cannot have a server trace")
        seen_traces.add(request_id)
        bucket = server[by_id[request_id]["outcome"]]
        bucket["duration"].append(trace["duration_ns"])
        per_request = Counter()
        for stage in trace["stages"]:
            per_request[stage["name"]] += stage["end_ns"] - stage["start_ns"]
        bucket["unattributed"].append(trace["duration_ns"] - sum(per_request.values()))
        for name, duration in per_request.items():
            bucket["stages"][name].append(duration)

    seconds = (window["end_ns"] - window["start_ns"]) / 1_000_000_000
    warnings = []
    if manifest["evidence_kind"] == "synthetic":
        warnings.append("Synthetic fixture: arithmetic/plumbing evidence only, not model or service performance.")
    if counts["success"] < 2000:
        warnings.append("Fewer than 2000 successful observations: treat the p95 as exploratory.")
    if counts["timeout"] or counts["outstanding"] or counts["generator_drop"]:
        warnings.append("Success latency omits censored/unfinished/unsent work; inspect all outcome counts and send lag.")
    if seen_traces and len(seen_traces) < len(requests) - counts["generator_drop"]:
        warnings.append("Server traces cover only part of the sent cohort; inspect each stage's sample count.")
    configuration = manifest.get("reproducibility", {}).get("configuration")
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest() if configuration is not None else None
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "evidence_kind": manifest["evidence_kind"],
        "reproducibility": manifest.get("reproducibility", {}),
        "reproducibility_missing": missing,
        "configuration_sha256": configuration_hash,
        "quantile_method": {"p50": "median", "p95": "nearest_rank"},
        "window": window,
        "observation_end_ns": manifest["observation_end_ns"],
        "timeout_ns": manifest["timeout_ns"],
        "counts": {
            "offered": len(requests), "sent": len(requests) - counts["generator_drop"],
            **dict(counts), "success_completed_in_window": in_window_success,
        },
        "offered_requests_per_second": len(requests) / seconds,
        "successful_completions_per_second_in_window": in_window_success / seconds,
        "send_lag_ms": distribution(send_lag),
        "client_elapsed_by_outcome_ms": {
            outcome: distribution(elapsed[outcome]) for outcome in ("success", "rejected", "error", "timeout")
        },
        "server_traces": {
            "count": len(seen_traces),
            "by_client_outcome": {
                outcome: {
                    "duration_ms": distribution(bucket["duration"]),
                    "unattributed_ms": distribution(bucket["unattributed"]),
                    "stages_ms": {name: distribution(values) for name, values in sorted(bucket["stages"].items())},
                }
                for outcome, bucket in sorted(server.items())
            },
        },
        "warnings": warnings,
    }


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number is invalid: {value}")


def load_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, parse_constant=_reject_constant))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize(load_json(args.manifest), load_jsonl(args.requests), load_jsonl(args.traces) if args.traces else ())
    except (OSError, ValueError) as error:
        print(f"invalid run: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
