"""Optional real-artifact checks: run after build, never download/train in tests."""

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from benchmarks.timing import validate_trace
from compact_vision.common import read_json
from compact_vision.preprocessing import decode_image, preprocess
from compact_vision.runtime import make_backend
from compact_vision.service import create_app

ARTIFACTS = Path("models/demo")
pytestmark = pytest.mark.skipif(not (ARTIFACTS / "model.fp32.onnx").exists(), reason="build real local artifacts first")


def test_real_export_and_service_predictions_match_for_the_same_encoded_input():
    rng = np.random.default_rng(53)
    stream = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (51, 37, 3), dtype=np.uint8)).save(stream, format="PNG")
    encoded = stream.getvalue()
    inputs = preprocess(decode_image(encoded))
    reference = make_backend(ARTIFACTS, "torch").predict(inputs)
    exported = make_backend(ARTIFACTS, "fp32").predict(inputs)
    config = read_json("configs/demo.json")
    np.testing.assert_allclose(exported, reference, atol=config["parity_atol"], rtol=config["parity_rtol"])
    wrong_input = np.ascontiguousarray(np.asarray(decode_image(encoded).resize((224, 224)), dtype=np.float32).transpose(2, 0, 1)[None] / 255)
    wrong_logits = make_backend(ARTIFACTS, "fp32").predict(wrong_input)
    assert not np.allclose(wrong_logits, reference, atol=config["parity_atol"], rtol=config["parity_rtol"])
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["runtime"]["providers"] == ["CPUExecutionProvider"]
        response = client.post("/predict", content=encoded, headers={"content-type": "image/png", "x-request-id": "real-artifact-test"})
        assert response.status_code == 200
        assert response.json()["predictions"][0]["index"] == int(exported.argmax(1)[0])
        trace = json.loads(response.headers["x-compact-trace"])
        validate_trace(trace)
        assert {"decode", "preprocess", "model", "executor_wait", "serialize"} <= {stage["name"] for stage in trace["stages"]}
        invalid = client.post("/predict", content=b"corrupt", headers={"content-type": "image/png"})
        assert invalid.status_code == 422
        validate_trace(json.loads(invalid.headers["x-compact-trace"]))
