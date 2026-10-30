# Ownership and improvement pass

## Technical question

For the existing classifier and its real input contract, where does batch-one CPU request time go, and does one alternative ONNX Runtime intra-op thread setting improve the relevant operating point without changing correctness or hiding overload?

This is a proposed contribution. The completed baseline is not in this checkout, so source-specific scope and effort remain unresolved. If it already contains this study, recover that evidence and choose a different justified gap before changing code.

## Ownership boundaries

| Layer | Status and attribution |
| --- | --- |
| Upstream | TorchVision models/weights/transforms and ONNX Runtime inference/quantization infrastructure; exact adopted versions/examples remain to be identified |
| Existing personal baseline | Reported complete by the owner on another computer; implementation and saved experiment configuration not inspected here |
| This checkout | Repository setup, operating documentation and locally ignored beginner curriculum |
| Proposed technical contribution | Missing stage measurements, controlled thread experiment, reproducible decision and explanation; not implemented or measured yet |

Changing names, adding a wrapper or listing more frameworks is not the contribution. The contribution earns its place through an unanswered practical question, a fair experiment and a technical decision supported by inspectable evidence. Mixed or negative results are acceptable.

## Milestones and acceptance

| Milestone | Work | Acceptance |
| --- | --- | --- |
| 0. Recover baseline | Preserve original history/local changes; read instructions; bind model/data/transforms/export/quantizer/service/benchmark/environment/attribution to real locations | Actual source/artifact map and one useful saved baseline record, with unknowns retained; estimate effort only now |
| 1. Trace correctness | Follow a permitted image from bytes to class output; validate class order, deterministic evaluation/serving transform and export parity | Actual shapes/dtypes/values recorded; fixed artifact/input identity and justified FP32 tolerance; no unexplained mismatch |
| 2. Explain stages | Reuse or extend the existing runner and minimal timing hooks; distinguish decode, preprocess, runtime, response and observed wait | Per-request intervals reconcile with documented residuals; instrumentation overhead checked; failures retained |
| 3. Select one alternative | Development-only sweep of current intra-op setting versus 1, 2, 4 where supported; hold other thread pools, service workers, batch, input and model fixed | One candidate selected by a declared rule, then frozen before final evaluation |
| 4. Paired evaluation | Current versus candidate on the same host, inputs and traffic; repeated measurements with outcome accounting | Correctness preserved; p50/p95 and successful throughput accompanied by variation, timeouts/errors/rejections and raw evidence; keep/change decision |
| 5. Explain independently | One-page explainer, architecture from memory, reproduce a metric and answer the twelve questions aloud | Clear attribution, justified decisions, three concrete failure modes and next experiment; private answers remain private |

## Evaluation contract

- Reuse existing artifacts and targeted evidence; do not automatically rebuild, retrain or repeat the entire baseline.
- Compare identical model/input/configuration lineage on one named host. Match batch and CPU budgets and explicitly set provider/options. Keep native/container/emulated results separate.
- Preserve data separation: training for fitting, representative training-only inputs for calibration, development inputs for choosing settings, untouched test inputs for final quality. Record inherited limitations; do not silently reshuffle splits.
- Use labeled classification quality only if that is the actual task: correct/total, top-1 accuracy and relevant per-class failures. Declare acceptable INT8 quality loss before final scoring. A thread change should not introduce unexplained prediction differences.
- FP32 parity uses identical preprocessed tensors, eval mode, output order, shape/finiteness and numerical errors with a justified frozen tolerance. Matching top-1 labels alone is insufficient.
- Model-only timing excludes decoding/preprocessing/session construction. Request timing uses original encoded bytes. Measure lazy decoding where pixels are materialized. Do not label client-minus-server residual or subtracted p95s as queue time.
- Proposed model protocol: 50 untimed warmups, 1,000 timed batch-one calls, three fresh-process repeats; verify adequacy on development inputs before freezing. These are protocol proposals, not measured counts.
- Final HTTP matrix is one current/candidate pair at three fixed offered rates below/near/modestly above the development capacity knee. Record actual send schedule, send lag, in-flight cap, timeout/retry/drain policy and generator placement. Closed-loop concurrency alone cannot establish offered-load capacity.
- Proposed request protocol: aim for 2,000 scheduled arrivals and at least 60 seconds per condition/repeat, with a 180-second cap and bounded drain; report realized counts and label sparse p95 estimates exploratory. Three paired repeats, alternating order. No p99 claims on sparse data.
- Count offered, sent, generator-dropped, successful, rejected, errored, timed-out and outstanding requests. Successful response latency must accompany failures. Late completion does not erase a client timeout. Specify throughput window boundaries and distinguish service success from classification accuracy.
- Save raw observations and each repeat. Report median/range and suitably bounded uncertainty rather than a universal gain. Keep separate backend-context comparisons from the thread experiment's causal claim.
- Before selecting an INT8 path inspect the serialized graph/IR/opsets, installed runtime/quantizer, provider placement, supported operators/types and hardware. ORT generally recommends static quantization for CNNs; both dynamic and static need not work well here. [Official guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Unsupported or slower INT8 is an acceptable finding.

## Boundaries and stopping rule

Minimum scope: one existing task/model, one host, batch one, stage evidence and one threading comparison. Optional static INT8 work is a separate stretch only after actual support and a missing baseline capability are established.

Excluded: retraining, architecture search, new dataset collection, GPU port, dynamic batching, new scheduler, autoscaling, distributed deployment, dashboards and other projects. The 10×/100× questions teach reasoning; they are not required capacity experiments.

Stop after one development sweep and one frozen paired evaluation with a defensible decision and explanation. Allow a targeted repair/repeat for a demonstrated measurement fault. Do not retune on final results or chase a promised percentage. Missing baseline assets stop dependent implementation; useful independent documentation may proceed.

## Planned public evidence

Adapt the existing repository's tooling and directories after transfer. Keep timing definitions, exact commands/configuration, attribution and selected permitted result records versioned. No results are filled in yet.

| Model/config/run | Input/artifact/config identity | Quality correct/N | Model p50/p95 | Client p50/p95 | Successful req/s | Offered/sent/outcome counts | Repeat variation | Decision/evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Current | | | | | | | | |
| Candidate | | | | | | | | |
