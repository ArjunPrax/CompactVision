# Measurement helpers: schema 1

This independently written component validates recorded client outcomes and sequential server-stage traces. It does not load a model, generate traffic or replace the completed baseline. Integration into that baseline and the controlled threading comparison remain pending source/artifact access.

The implementation uses Python's standard library. It has been checked with Python 3.14.3; no dependencies were installed. Run commands from the repository root.

## Check the implementation

This command runs contract/arithmetic tests, including invalid records. It performs no model inference or load test:

```sh
python3 -m unittest discover -s tests -v
```

This command reads deliberately fabricated fixture records, validates them and prints JSON to the terminal. It does not measure a live system or create result files:

```sh
python3 -m benchmarks.report tests/fixtures/synthetic/manifest.json tests/fixtures/synthetic/requests.jsonl --traces tests/fixtures/synthetic/traces.jsonl
```

The output preserves `evidence_kind: synthetic` and warns that these are accounting/plumbing checks. None of its times, rates or placeholder metadata are project performance results.

## Files and interfaces

| Component | Responsibility |
| --- | --- |
| [timing.py](../benchmarks/timing.py): `Trace.stage`, `Trace.finish` | Capture sequential stage intervals relative to one request's local monotonic origin |
| [report.py](../benchmarks/report.py): `summarize` | Validate a single run/repeat and aggregate outcomes, latency, send lag and optional server-stage evidence |
| `report.py`: module CLI | Read a manifest JSON and request JSONL, plus optional trace JSONL; print a JSON report, exit 2 on invalid input |
| [test_measurement.py](../tests/test_measurement.py) | Hand-calculated accounting, deadline/window, trace and validation failure cases |

JSON stores structured values. JSONL stores one JSON object per line. Blank JSONL lines are ignored. Nonfinite JSON values are rejected. Each request ID appears once per run; schema 1 does not support retries or multi-attempt joins. Reports do not combine repeats/configurations or calculate a speedup.

## Manifest

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `run_id` | Nonempty run/repeat identifier |
| `evidence_kind` | Explicitly `synthetic` or `measured`; never inferred from filenames |
| `client_clock` | `monotonic_offsets_ns`: client schedule/send/finish/window use offsets from one shared local origin |
| `window.start_ns`, `window.end_ns` | Integer nanoseconds defining positive-duration half-open arrival/completion window `[start,end)` |
| `observation_end_ns` | End of observation/drain; at least the window end |
| `scheduled_requests` | Nonnegative count; exactly one request row must exist for each scheduled arrival, including unsent ones |
| `timeout_ns` | Positive per-request timeout from actual send; schema 1 uses the same timeout for every request |
| `reproducibility` | Metadata declarations described below; required in full for a `measured` run |

All timestamps are nonnegative integers; booleans/floats are rejected as timestamps/counts. A client timestamp cannot be compared directly to a separately originated server trace offset.

Required reproducibility declarations for measured runs:

- `model_sha256`, `inputs_sha256`: lowercase 64-character SHA-256 digests identifying the actual model artifact and ordered input manifest. For external model data, define a manifest covering all files and identify it consistently; a digest cannot identify omitted files.
- `configuration`: `batch_size`, `service_workers` (positive integer counts); objects `runtime_threads`, `other_thread_pools`, `runtime_options`. Include actual intra/inter-op/execution settings and options as applicable. An empty options/pools object explicitly represents no applicable settings; do not omit active pools.
- `environment`: `hardware`, `os`, `architecture`, `python`, `runtime`, `provider`. Record actual CPU/RAM allocation, exact build/provider/options and native/container/emulated context in these declarations. Additional keys are preserved.
- `code_revision`: a nonempty source identifier. Include the relevant patch/untracked code identities alongside it for a dirty run; the reporter cannot recover absent provenance.
- `warmup_calls`: nonnegative count; `offered_rate_rps`: finite positive intended rate; `max_in_flight`: positive cap; `retry_policy`: `disabled`.
- `timing_boundaries`: descriptions of `client` and `server` boundaries. State uninstrumented/excluded work honestly, including body read, dispatch, OS backlog or serialization as relevant. Add actual stage definitions here too.

The reporter checks field presence/basic types and hashes the canonical configuration object for identification. It cannot authenticate declarations, inspect real provider placement, validate preprocessing/parity/task quality, establish representativeness or certify that a run was measured. Those remain integration/evaluation responsibilities. Synthetic fixtures may omit metadata; missing fields stay visible rather than being filled with guessed hardware or model values.

## Client request rows

Each row has `request_id`, `scheduled_ns`, `sent_ns`, `finished_ns` and `outcome`. All scheduled arrivals must lie in `[window.start_ns, window.end_ns)`, even if the client sends late during observation/drain. Actual send must be at/after the scheduled arrival and no later than observation end.

| Outcome | Required event shape |
| --- | --- |
| `success` | Sent; valid complete response strictly before `sent_ns + timeout_ns` and at/before observation end. This means service response success, not a correct predicted class. |
| `rejected` | Sent; refusal received strictly before the deadline; a late refusal remains a client timeout. |
| `error` | Sent; failure completed strictly before the deadline; errors are not silently removed. |
| `timeout` | Sent; client deadline expired; finish/detection timestamp at/after the deadline and at/before observation end. A later server completion must not turn this into success. |
| `outstanding` | Sent, no client finish yet, and observation ends strictly before its deadline. Once the deadline is reached, classify it as timeout. |
| `generator_drop` | Scheduled but never sent; both send and finish are null/absent. No server trace can belong to it. |

A request with a response timestamp exactly at its deadline is a timeout under this explicit convention. A `timeout` elapsed duration records the observed deadline/detection interval; it is censored, not the unknown eventual server completion time.

The reporter rejects missing arrivals, duplicate IDs, invalid time ordering and late responses falsely marked successful. It cannot detect a dishonest scheduled-count declaration; use the generator's complete scheduled manifest.

## Server trace rows

Each trace has `request_id`, `duration_ns`, and `stages`, a list of `{name,start_ns,end_ns}`. The stage offsets start at the trace's own local origin. Stages must be ordered, nonoverlapping and inside the trace duration. Stage names can repeat: their durations are summed **per request** before calculating a stage distribution. This avoids treating two pieces of one decode as two independent request samples.

`Trace(request_id)` uses `time.perf_counter_ns()` by default. The `stage(name)` context manager records elapsed work even when the wrapped operation raises, then lets that exception propagate. Call `finish()` once after handling the exception if the integration needs the failed request's trace. One `Trace` belongs to one sequential request path: it is not shared between tasks/threads, does not support nested/overlapping stages, and rejects use after finishing. An async/parallel service needs an explicit integration decision about interval representation rather than this helper silently double-counting overlapping work.

Keep the actual decoder's materialization inside `decode`; place normalization/copies inside the stated preprocessing boundary; exclude session construction from steady-state `model` time. Stage names alone cannot enforce these semantics. Measure instrumentation overhead on the real development workload before final comparisons. These hooks have not been placed into the unavailable service yet.

Trace IDs must match sent client requests, and each can have only one server trace in schema 1. Optional server traces are grouped by **client outcome**, so timeout/error traces cannot silently become success-stage evidence. Missing server traces remain missing; inspect per-stage sample counts. No server-received count is inferred from client outcomes or partial tracing.

## Report definitions

- Counts retain all scheduled arrivals: `offered = sent + generator_drop`; `sent = success + rejected + error + timeout + outstanding`.
- `offered_requests_per_second` is scheduled arrivals divided by window seconds. The declared intended rate remains in metadata; send lag exposes generator schedule slippage.
- `successful_completions_per_second_in_window` counts this scheduled cohort's successful client completions in `[start,end)`, divided by window seconds. A completion after the window during drain remains cohort success but is excluded from this rate. Warmup/prior-cohort rows are excluded from this schema's scheduled cohort; this is not automatically a stationary capacity estimate.
- Client elapsed distributions are separated by outcome. Success p50/p95 never silently pool rejections, errors or censored timeouts. Empty samples have null quantiles, not zero latency. Send lag is `sent - scheduled` for sent requests only; unsent requests still appear in counts.
- p50 is the arithmetic median; p95 uses nearest rank: sort N values and choose one-based rank `ceil(0.95*N)`. The method is recorded. No p99 is reported. The warning below 2,000 successes follows the proposed experiment's sample aim; that threshold is not a statistical guarantee. Every per-stage distribution has its own count, potentially much smaller.
- Within each sequential trace, `unattributed = trace duration - sum(stage durations)`. This includes uncovered local work and timer gaps; it is **not automatically queue time**. No client/server timestamp subtraction or percentile subtraction is performed, and stage p95s are not summed.

Use the improvement plan's fixed-model/current-versus-candidate, paired-repeat evaluation later. This reporting layer alone does not establish a speedup, identify a bottleneck or complete the ownership contribution.
