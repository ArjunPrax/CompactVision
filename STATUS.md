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
- 32 tests passed, including original accounting tests, preprocessing equivalence, decoder/API contracts, split guards and traffic timeout/drop accounting. Ruff passed. A dependency emits a TestClient deprecation warning; assertions passed.
- Public dataset download/training is in progress. Actual trained artifacts, export parity, quality and live timing are not yet verified; no gain is claimed.
- Docker is not installed on this host, so image build/run is unverified.

## Next action

Finish the archive download and CPU head training; execute export/parity and static quantization support checks before quality/timing. Run the frozen comparison and record actual outcomes and limitations. Bind private guides to this rebuilt source and its observed input trace. Preserve Git history and keep guides/data/weights untracked.
