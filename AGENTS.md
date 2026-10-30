# Compact Vision working instructions

Read README.md, STATUS.md, IMPROVEMENT_PLAN.md and ATTRIBUTION.md before substantive work.

- Scope change, 30 October 2026: the owner explicitly states the other computer's work cannot be obtained and authorizes rebuilding everything here. Build a new, independently attributed CPU baseline; do not continue waiting for transfer. Keep the earlier completion report separate from new code/results. Do not invent old implementation details or measurements.
- Preserve existing work and original history. Before integrating the completed repository, inspect its instructions, remotes, HEAD, diffs and untracked files. Keep its commits available; do not overwrite it, force-push over history or flatten it into an unattributed source dump. Integrate this documentation with that history deliberately.
- Upstream libraries provide the new baseline's foundations. Keep upstream, the inaccessible historical implementation, the rebuilt baseline and new improvements distinct.
- The rebuilt baseline may train a small classification head on public images, export/verify ONNX and test a supported static INT8 path. The ownership improvement remains stage-resolved CPU latency diagnosis plus one controlled threading comparison. No serving platform or guaranteed gain.
- Keep public evidence reproducible: model/input/config hashes, code revision and relevant patch, environment/hardware, timing boundaries, warmup, batch/thread/worker settings, traffic schedule and all request outcomes.
- Test correctness before timing. Separate FP32 export parity from INT8 task-quality tolerance; separate calibration/development/test data; inspect actual graph/runtime/provider/operator/hardware support.
- Preserve the root-only `/.learning/` ignore rule. Before committing, run `git check-ignore -v .learning/README.md` and `git ls-files -- .learning`; the latter must be empty. Ignore rules do not untrack files. Keep private notes out of Git and maintain a separate private backup.
- Lessons assume no prior knowledge: define words and notation, show a worked example, give a small exercise, provide progressive hints separately and answer checks separately. Explain all commands before asking the learner to run them. Leave personal answers and predictions for the learner.
- Mark conceptual/toy examples, deliberately injected bugs, observed project facts and source bindings clearly. Update the lessons against this rebuilt implementation; performance claims require actual measurements.
- Use public sources and independently written work. No employer/lab/private research assets or unreleased results in this project or its learning folder.
- Keep STATUS.md as the canonical current status/next action. Add test and lint commands only after actual project tooling is inspected; do not install a new stack merely to match another project.
- Repository visibility starts private. Do not publish a website, change visibility or release data/weights without explicit authorization.

## Current validation tooling

The measurement helpers use the Python standard library. Verified with Python 3.14.3; no dependencies were installed. Run `python3 -m unittest discover -s tests -v` from the root. Tests/fixtures validate accounting, schema and timer behavior only. `docs/measurement.md` defines the interfaces and synthetic report command. There is no model/service smoke command or established lint tool until the actual baseline tooling is inspected.
