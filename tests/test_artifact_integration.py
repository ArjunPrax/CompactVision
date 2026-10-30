"""Optional real-artifact checks: run after build, never download/train in tests."""

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from benchmarks.timing import validate_trace
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
    np.testing.assert_allclose(exported, reference, atol=1e-5, rtol=1e-4)
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
