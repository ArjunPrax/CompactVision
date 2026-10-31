# Compact Vision working instructions

Read README.md, STATUS.md, IMPROVEMENT_PLAN.md and ATTRIBUTION.md before substantive work.

- Write the README and public technical documentation from the developer's perspective. Describe the system, design decisions, attribution, measurements and limitations. Keep agent/owner authorization language, conversation history and handoff logistics out of public project documentation.

- Present CompactVision as the developer's image-classification service and CPU inference study. Describe implemented behavior and recorded results; do not invent implementation details or measurements.
- Preserve existing work and Git history. Inspect applicable instructions, remotes, HEAD, diffs and untracked files before changes. Do not overwrite local work or force-push over history.
- Attribute upstream models, pretrained weights, libraries and algorithms. Explain the project's application, correctness checks, instrumentation and controlled experiments as its engineering contributions.
- The baseline trains a small classification head on public images, exports/verifies ONNX and evaluates static INT8. The performance study diagnoses CPU latency by stage and compares thread settings. Keep further work bounded; do not promise gains or expand into a serving platform.
- Keep public evidence reproducible: model/input/config hashes, code revision and relevant patch, environment/hardware, timing boundaries, warmup, batch/thread/worker settings, traffic schedule and all request outcomes.
- Test correctness before timing. Separate FP32 export parity from INT8 task-quality tolerance; separate calibration/development/test data; inspect actual graph/runtime/provider/operator/hardware support.
- Preserve the root-only `/.learning/` ignore rule. Before committing, run `git check-ignore -v .learning/README.md` and `git ls-files -- .learning`; the latter must be empty. Ignore rules do not untrack files. Keep private notes out of Git and maintain a separate private backup.
- Lessons assume no prior knowledge: define words and notation, show a worked example, give a small exercise, provide progressive hints separately and answer checks separately. Explain all commands before asking the learner to run them. Leave personal answers and predictions for the learner.
- Mark conceptual/toy examples, deliberately injected bugs, observed project facts and source bindings clearly. Keep lessons aligned with the implementation; performance claims require actual measurements.
- Use public sources and independently written work. No employer/lab/private research assets or unreleased results in this project or its learning folder.
- Keep STATUS.md as the canonical current status/next action. Add test and lint commands only after actual project tooling is inspected; do not install a new stack merely to match another project.
- The repository is public. Do not publish a website, change visibility or release data/weights without explicit authorization.

## Current validation tooling

Use the locked Python 3.12 environment: `uv sync --frozen`, `uv run pytest -q`, `uv run ruff check .`. Full tests use isolation doubles plus an optional real-artifact integration check after building. The standard-library-only subset is `python3 -m unittest discover -s tests -p test_measurement.py -v`. Do not call synthetic tests accuracy/latency evidence.

`uv run compact-vision build` downloads public data/weights and runs the head-training/export/validation pipeline in a fresh artifact directory. `trace`, `benchmark`, `study` and explicit `evaluate --split test` produce the actual evidence. Inspect `docs/evaluation.md` and the saved frozen protocol before interpreting reports. Keep default serving FP32 unless a separate justified deployment decision changes it. The retained INT8 experiment is classifier-only with a float convolution backbone, and must be labeled as partial INT8.
