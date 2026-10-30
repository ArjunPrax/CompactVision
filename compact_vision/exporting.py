"""FP32 export, development parity and one supported static CNN quantizer."""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch

from .common import environment, object_hash, provenance, sha256, write_json
from .data import dataset
from .model import load_model
from .preprocessing import CONTRACT, preprocess
from .runtime import OnnxBackend


def graph_info(path):
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    return {
        "sha256": sha256(path), "bytes": Path(path).stat().st_size, "ir_version": graph.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in graph.opset_import},
        "operators": dict(Counter(node.op_type for node in graph.graph.node)),
        "initializer_dtypes": dict(Counter(onnx.TensorProto.DataType.Name(item.data_type) for item in graph.graph.initializer)),
    }


def export_fp32(config, manifest):
    artifacts = Path(config["artifacts"])
    model = load_model(artifacts / "checkpoint.pt")
    torch.set_num_threads(config["baseline_threads"])
    path = artifacts / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(path), input_names=["images"], output_names=["logits"],
                          opset_version=17, dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                          dynamo=False, external_data=False)
    session = OnnxBackend(path, threads=config["baseline_threads"])
    source = dataset(config["data_root"], True)
    errors, failures, disagreements = [], 0, 0
    for index in manifest["indices"]["validation"][:32]:
        inputs = preprocess(source[index][0])
        with torch.inference_mode():
            reference = model(torch.from_numpy(inputs)).numpy()
        actual = session.predict(inputs)
        difference = np.abs(actual - reference)
        failures += int(np.sum(difference > config["parity_atol"] + config["parity_rtol"] * np.abs(reference)))
        disagreements += int(reference.argmax(1)[0] != actual.argmax(1)[0])
        errors.append({"source_index": index, "max_absolute_error": float(difference.max()), "mean_absolute_error": float(difference.mean()),
                       "max_relative_error_with_1e-8_floor": float((difference / np.maximum(np.abs(reference), 1e-8)).max())})
    # Exercise the exported dynamic batch dimension too.
    pair = np.concatenate([preprocess(source[i][0]) for i in manifest["indices"]["validation"][:2]])
    with torch.inference_mode():
        np.testing.assert_allclose(session.predict(pair), model(torch.from_numpy(pair)).numpy(), atol=config["parity_atol"], rtol=config["parity_rtol"])
    report = {"artifact": graph_info(path), "preprocessing": CONTRACT, "input_signature": "float32 NCHW [N,3,224,224], N>=1",
              "output_signature": "float32 [N,10] logits, CIFAR-10 class order", "corpus": "first 32 frozen validation indices",
              "n": len(errors), "atol": config["parity_atol"], "rtol": config["parity_rtol"],
              "element_tolerance_failures": failures, "top1_disagreements": disagreements, "per_input": errors,
              "dynamic_batch_two_passed": True, "checkpoint_sha256": sha256(artifacts / "checkpoint.pt"),
              "environment": environment(), "code": provenance()}
    write_json(artifacts / "parity.json", report)
    if failures:
        raise RuntimeError("FP32 parity failed; stop before quantization or timing")
    print(f"Exported ONNX FP32; parity passed on {len(errors)} inputs and batch two", flush=True)
    return report


class CalibrationReader(CalibrationDataReader):
    def __init__(self, root, indices):
        self.source = dataset(root, True)
        self.indices = indices
        self.rewind()

    def get_next(self):
        index = next(self.iterator, None)
        return None if index is None else {"images": preprocess(self.source[index][0])}

    def rewind(self):
        self.iterator = iter(self.indices)


def quantize(config, manifest):
    artifacts = Path(config["artifacts"])
    prepared = artifacts / "model.preprocessed.onnx"
    quant_pre_process(str(artifacts / "model.fp32.onnx"), str(prepared), skip_symbolic_shape=True)
    output = artifacts / "model.int8.onnx"
    quantize_static(str(prepared), str(output), CalibrationReader(config["data_root"], manifest["indices"]["calibration"]),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    # Execution and provider placement must be checked, not inferred from conversion success.
    backend = OnnxBackend(output, threads=config["baseline_threads"], profile_prefix=artifacts / "int8-profile")
    source = dataset(config["data_root"], True)
    backend.predict(preprocess(source[manifest["indices"]["validation"][0]][0]))
    profile_path = backend.session.end_profiling()
    from .common import read_json
    events = read_json(profile_path)
    kernels = Counter()
    providers = Counter()
    for event in events:
        if event.get("cat") == "Node":
            args = event.get("args", {})
            kernels[args.get("op_name", "unknown")] += 1
            providers[args.get("provider", "unknown")] += 1
    report = {"status": "supported", "format": "static S8S8 QDQ", "method": "MinMax", "per_channel_weights": True,
              "requested_operators": ["Conv", "Gemm", "MatMul"], "artifact": graph_info(output),
              "float_source_sha256": sha256(artifacts / "model.fp32.onnx"),
              "calibration_source": "training subset only", "calibration_indices": manifest["indices"]["calibration"],
              "calibration_manifest_sha256": object_hash(manifest["indices"]["calibration"]),
              "profile_kernel_events": dict(kernels), "profile_provider_events": dict(providers),
              "runtime": backend.info, "environment": environment(), "code": provenance()}
    write_json(artifacts / "quantization.json", report)
    print(f"Static INT8 executes; actual profiled kernel events: {dict(kernels)}", flush=True)
    return report
