# Compact Vision Inference Service

A personal study of compact vision inference, correctness and CPU request latency, using [TorchVision](https://github.com/pytorch/vision) and [ONNX Runtime](https://github.com/microsoft/onnxruntime) as upstream foundations.

## Current state

This repository contains the ownership-pass documentation and independently written measurement helpers: sequential request-stage timing and validated outcome/latency reporting. The owner reports a completed baseline on another computer; that implementation, checkpoints, dataset manifests and result logs have not been transferred into this checkout. There is no runnable classifier or inference service here yet. This does not change the reported completion of the original work.

See [STATUS.md](STATUS.md) for verified progress and the next action. Do not interpret repository setup or learning exercises as measured inference results.

The [measurement guide](docs/measurement.md) documents the implemented interfaces, record schema and commands. Its synthetic fixtures test arithmetic and accounting only; they are not performance results. The helpers use the Python standard library and require no dependency installation.

## Intended contribution

Explain where a real request spends time—image decoding, preprocessing, model execution, postprocessing and observable waiting—then compare one explicit CPU threading alternative against the baseline configuration. Preserve the model, task and input contract so the comparison answers a focused engineering question.

The outcome may be a configuration change or evidence that the existing configuration should be retained. A smaller model or faster forward pass alone does not establish better request latency.

The [improvement plan](IMPROVEMENT_PLAN.md) defines ownership boundaries, acceptance checks, evaluation rules and exclusions. [ATTRIBUTION.md](ATTRIBUTION.md) separates upstream foundations from existing personal work and the proposed new contribution.

## Learning and reproducibility

Private beginner lessons live locally in `.learning/`, ignored by `/.learning/` and backed up separately. They start with programming, images and model fundamentals, then cover preprocessing, transfer learning, ONNX, quantization and performance measurement. They do not accompany a normal clone.

Public technical definitions, reproducible evaluation code/configuration and selected permitted results will remain outside that folder. Until the baseline is available, the lessons distinguish conceptual examples from unverified project details. No new accuracy, speedup or capacity result has been measured in this repository.
