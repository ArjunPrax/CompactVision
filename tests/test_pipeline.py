"""Meaningful input/API/traffic checks, independent of downloaded weights/data."""

import asyncio
import io
import json

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image
import pytest
from torchvision.models import MobileNet_V3_Small_Weights

from benchmarks.timing import validate_trace
from compact_vision.common import read_json
from compact_vision.data import development_indices, validate_splits
from compact_vision.experiment import traffic
from compact_vision.preprocessing import decode_image, preprocess
from compact_vision.runtime import PredictionError
from compact_vision.service import create_app


def image_bytes(size=(31, 19), mode="RGB"):
    stream = io.BytesIO()
    Image.new(mode, size, 127).save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("size", [(32, 32), (301, 107), (107, 301), (257, 256)])
def test_preprocessing_matches_upstream_weight_transform_on_nonsquare_inputs(size):
    rng = np.random.default_rng(9)
    image = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    reference = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()(image).numpy()[None]
    np.testing.assert_allclose(preprocess(image), reference, atol=1e-6, rtol=1e-6)
    assert preprocess(image).flags.c_contiguous


def test_decode_materializes_rgb_and_rejects_corrupt_oversize_and_animated_images():
    decoded = decode_image(image_bytes(mode="L"))
    assert decoded.mode == "RGB" and np.asarray(decoded).shape == (19, 31, 3)
    for content, limits in [(b"broken", {}), (b"", {}), (image_bytes(), {"max_bytes": 10}), (image_bytes(), {"max_pixels": 1})]:
        with pytest.raises(ValueError):
            decode_image(content, **limits)
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, format="PNG", save_all=True, append_images=[Image.new("RGB", (2, 2), "blue")])
    with pytest.raises(ValueError):
        decode_image(stream.getvalue())


def test_split_leakage_checks_respect_source_split_and_calibration_subset():
    validate_splits({"train": [1, 2], "validation": [3], "calibration": [2], "test": [1]})
    for splits in (
        {"train": [1, 1], "validation": [3], "calibration": [1], "test": [1]},
        {"train": [1], "validation": [1], "calibration": [1], "test": [1]},
        {"train": [1], "validation": [3], "calibration": [3], "test": [1]},
    ):
        with pytest.raises(ValueError):
            validate_splits(splits)


def test_development_parity_corpus_covers_classes_before_repeated_examples():
    from types import SimpleNamespace
    source = SimpleNamespace(targets=[0, 0, 1, 1, 2, 2])
    manifest = {"indices": {"validation": [0, 1, 2, 3, 4, 5]}}
    selected = development_indices(manifest, source, 4)
    assert selected == [0, 2, 4, 1]


class FakeEngine:
    """API isolation double; no claim of actual model inference."""
    constructions = 0

    def __init__(self, *args, **kwargs):
        from types import SimpleNamespace
        FakeEngine.constructions += 1
        self.classes = [f"fixture-{i}" for i in range(10)]
        self.backend = SimpleNamespace(model_sha256="a" * 64, info={"providers": ["fixture"]})

    def predict_bytes(self, content, request_id):
        from time import perf_counter_ns
        origin = perf_counter_ns()
        if content == b"broken":
            raise PredictionError("fixture decode error", {"request_id": request_id, "duration_ns": 0, "stages": []}, origin, 422)
        return {"request_id": request_id, "predictions": [{"label": "fixture"}]}, {"request_id": request_id, "duration_ns": 0, "stages": []}, origin


def test_api_session_lifetime_errors_and_real_observable_trace_contract():
    FakeEngine.constructions = 0
    app = create_app(engine_factory=FakeEngine)
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        body = client.get("/openapi.json").json()["paths"]["/predict"]["post"]["requestBody"]
        assert body["content"]["image/png"]["schema"]["format"] == "binary"
        for _ in range(2):
            response = client.post("/predict", content=image_bytes(), headers={"content-type": "image/png", "x-request-id": "test-image"})
            assert response.status_code == 200
            trace = json.loads(response.headers["x-compact-trace"])
            validate_trace(trace)
            assert "executor_wait" in [stage["name"] for stage in trace["stages"]]
        assert FakeEngine.constructions == 1
        assert client.post("/predict", content=b"broken", headers={"content-type": "image/png"}).status_code == 422
        assert client.post("/predict", content=b"text", headers={"content-type": "text/plain"}).status_code == 415
        assert client.post("/predict", content=b"a", headers={"content-type": "image/png", "x-request-id": "bad/id"}).status_code == 400


def test_open_arrival_generator_retains_malformed_responses_as_errors():
    async def respond(request):
        return httpx.Response(200, json=[])
    config = read_json("configs/demo.json")
    records, traces, observed, expected = asyncio.run(traffic("http://fixture", config, [b"fixture"], rate=20, duration=0.1,
                                                             warmup=0, transport=httpx.MockTransport(respond)))
    assert expected == len(records) == 2 and observed >= 100_000_000
    assert all(row["outcome"] == "error" for row in records)
    assert not traces


def test_open_arrival_timeouts_and_client_cap_keep_all_offered_arrivals():
    async def slow(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={})
    config = {**read_json("configs/demo.json"), "http_timeout_seconds": 0.03, "http_max_in_flight": 1}
    records, _, _, expected = asyncio.run(traffic("http://fixture", config, [b"fixture"], rate=100, duration=0.05,
                                                 warmup=0, transport=httpx.MockTransport(slow)))
    assert expected == len(records) == 5
    assert {row["outcome"] for row in records} == {"timeout", "generator_drop"}
    assert all(row["finished_ns"] >= row["sent_ns"] + 30_000_000 for row in records if row["outcome"] == "timeout")
