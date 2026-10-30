# Current status

Updated 30 October 2026.

## Scope decision

The owner explicitly authorized rebuilding everything here because the other computer's code cannot be obtained. Waiting for transfer is no longer a dependency. Earlier completion/approximate result reports remain historical user reports, separate from this rebuild.

## Implemented

- Locked Python 3.12 environment; MobileNetV3-Small frozen-feature extraction and a trainable ten-class CIFAR-10 head.
- Fixed stratified subsets, training-only calibration, exact-pixel duplicate protection across selected splits.
- Shared PIL/NumPy input contract checked against TorchVision on square/portrait/landscape images.
- CPU backends, export/parity, static INT8 support/profiling and labeled quality reporting.
- FastAPI lifecycle, one loaded session, bounded inputs, explicit errors and observable stage/executor-wait intervals.
- Fresh-process model timing, open-arrival traffic and one frozen paired threading comparison with all outcomes retained.
- Docker recipe, public reproduction documentation and private beginner curriculum.

## Verified so far

- Python 3.12.12, torch 2.15.0, torchvision 0.29.2, ONNX 1.23.2, ORT 1.31.1 installed and locked. CPUExecutionProvider is available and explicitly selected by the experiment code.
- 35 tests passed, including a real-checkpoint PyTorch/ONNX/API integration test, including original accounting tests, preprocessing equivalence, decoder/API contracts, split guards and traffic timeout/drop accounting. Ruff passed. A dependency emits a TestClient deprecation warning; assertions passed.
- Training completed. Validation-selected head: 1727/2000 (86.35%). Final FP32 parity passed 32 validation images and dynamic batch two after a documented development tolerance diagnosis. Original full-CNN INT8 failed quality (249/2000); one selective classifier-only correction scored 1720/2000 (86.0%), within the unchanged 2 pp validation guard. Convolution remains FP32. Live timing and final test scoring are next; no speed gain is claimed.
- Docker is not installed on this host, so image build/run is unverified.

## Next action

Freeze the validated methods, score the untouched test subset, and run the fresh-process model and paired HTTP studies. Record actual outcomes, limitations and the keep/change decision. Bind private guides to this rebuilt source and its observed input trace. Preserve Git history and keep guides/data/weights untracked.
