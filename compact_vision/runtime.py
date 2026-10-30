"""Explicit CPU backends and the shared, timed bytes-to-prediction path."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from benchmarks.timing import Trace

from .common import read_json, sha256
from .preprocessing import decode_image, preprocess


class OnnxBackend:
    def __init__(self, path, *, threads=1, profile_prefix=None):
        if threads < 1 or "CPUExecutionProvider" not in ort.get_available_providers():
            raise ValueError("a positive thread count and CPUExecutionProvider are required")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if profile_prefix:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "images" or inputs[0].type != "tensor(float)" or inputs[0].shape[1:] != [3, 224, 224]:
            raise ValueError("ONNX input does not match the rebuilt preprocessing contract")
        if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].shape[1:] != [10]:
            raise ValueError("ONNX output does not match the ten-class contract")
        self.model_sha256 = sha256(path)
        self.info = {"backend": "onnxruntime", "version": ort.__version__, "providers": self.session.get_providers(),
                     "intra_op_threads": threads, "inter_op_threads": 1, "execution_mode": "sequential",
                     "graph_optimization": "all", "allow_spinning": False}

    def predict(self, inputs):
        return self.session.run(["logits"], {"images": inputs})[0]


class TorchBackend:
    def __init__(self, path, *, threads=1):
        import torch
        from .model import load_model

        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.model = load_model(path)
        self.model_sha256 = sha256(path)
        self.info = {"backend": "pytorch", "version": torch.__version__, "device": "cpu", "intra_op_threads": threads,
                     "inter_op_threads": torch.get_num_interop_threads()}

    def predict(self, inputs):
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(inputs)).numpy()


def make_backend(artifacts, backend="fp32", threads=1, *, profile_prefix=None):
    artifacts = Path(artifacts)
    if backend == "torch":
        return TorchBackend(artifacts / "checkpoint.pt", threads=threads)
    if backend not in ("fp32", "int8"):
        raise ValueError("backend must be torch, fp32 or int8")
    return OnnxBackend(artifacts / f"model.{backend}.onnx", threads=threads, profile_prefix=profile_prefix)


class Engine:
    def __init__(self, artifacts, *, backend="fp32", threads=1, max_bytes=8_388_608, max_pixels=12_000_000, collect_stages=True):
        self.backend = make_backend(artifacts, backend, threads)
        self.classes = read_json(Path(artifacts) / "splits.json")["classes"]
        from .preprocessing import CONTRACT
        if len(self.classes) != 10 or len(set(self.classes)) != 10:
            raise ValueError("expected ten distinct class labels")
        if read_json(Path(artifacts) / "training.json")["preprocessing"] != CONTRACT:
            raise ValueError("training/serving preprocessing contract mismatch")
        self.max_bytes, self.max_pixels = max_bytes, max_pixels
        self.collect_stages = collect_stages

    def predict_bytes(self, encoded, request_id):
        trace = Trace(request_id, collect_stages=self.collect_stages)
        decoded = False
        try:
            with trace.stage("decode"):
                image = decode_image(encoded, max_bytes=self.max_bytes, max_pixels=self.max_pixels)
            decoded = True
            with trace.stage("preprocess"):
                inputs = preprocess(image)
            with trace.stage("model"):
                logits = self.backend.predict(inputs)[0]
            with trace.stage("postprocess"):
                if logits.shape != (len(self.classes),) or not np.isfinite(logits).all():
                    raise RuntimeError("backend returned invalid logits")
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                order = np.argsort(-probabilities)[:3]
                prediction = {"request_id": request_id, "model_sha256": self.backend.model_sha256,
                              "predictions": [{"index": int(i), "label": self.classes[i], "score": float(probabilities[i])} for i in order]}
        except Exception as error:
            status = 422 if isinstance(error, ValueError) and not decoded else 500
            raise PredictionError(str(error) if status == 422 else "inference failed", trace.finish(), trace.origin_ns, status) from error
        return prediction, trace.finish(), trace.origin_ns


class PredictionError(Exception):
    def __init__(self, message, trace, origin_ns, status_code):
        super().__init__(message)
        self.trace, self.origin_ns, self.status_code = trace, origin_ns, status_code
