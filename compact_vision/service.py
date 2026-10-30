"""One CPU session per process, bounded image input and observable executor wait."""

from contextlib import asynccontextmanager
import json
import os
import re
from time import perf_counter_ns
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .common import load_config
from .preprocessing import CONTRACT
from .runtime import Engine, PredictionError


def create_app(config_path=None, *, backend=None, threads=None, engine_factory=Engine):
    @asynccontextmanager
    async def lifespan(app):
        config = load_config(config_path or os.environ.get("COMPACT_VISION_CONFIG", "configs/demo.json"))
        artifacts = os.environ.get("COMPACT_VISION_ARTIFACTS", config["artifacts"])
        chosen_backend = backend or os.environ.get("COMPACT_VISION_BACKEND", "fp32")
        chosen_threads = threads or int(os.environ.get("COMPACT_VISION_THREADS", config["baseline_threads"]))
        tracing = os.environ.get("COMPACT_VISION_TRACING", "1") != "0"
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous_tokens = limiter.total_tokens
        limiter.total_tokens = 1  # One inference executor; wait is measured, not hidden.
        try:
            app.state.engine = engine_factory(artifacts, backend=chosen_backend, threads=chosen_threads,
                                              max_bytes=config["max_upload_bytes"], max_pixels=config["max_image_pixels"], collect_stages=tracing)
            app.state.config = config
            app.state.tracing = tracing
            yield
        finally:
            limiter.total_tokens = previous_tokens

    app = FastAPI(title="Compact Vision CPU inference", lifespan=lifespan)

    @app.get("/health")
    async def health():
        engine = app.state.engine
        return {"ready": True, "classes": engine.classes, "model_sha256": engine.backend.model_sha256,
                "runtime": engine.backend.info, "preprocessing": CONTRACT, "inference_executor_threads": 1, "trace_collection": app.state.tracing}

    @app.post("/predict", openapi_extra={"requestBody": {"required": True, "content": {
        "image/png": {"schema": {"type": "string", "format": "binary"}},
        "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
    }}})
    async def predict(request: Request):
        origin = perf_counter_ns()
        request_id = request.headers.get("x-request-id", uuid4().hex)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id):
            return JSONResponse({"error": "invalid request ID"}, status_code=400)
        if request.headers.get("content-type", "").split(";")[0].lower() not in ("image/png", "image/jpeg", "application/octet-stream"):
            return JSONResponse({"error": "send JPEG/PNG bytes as the raw request body"}, status_code=415)
        encoded = bytearray()
        async for chunk in request.stream():
            encoded.extend(chunk)
            if len(encoded) > app.state.config["max_upload_bytes"]:
                return JSONResponse({"error": "encoded image too large"}, status_code=413)
        body_end = perf_counter_ns()
        enqueued = perf_counter_ns()

        def execute():
            worker_start = perf_counter_ns()
            try:
                payload, trace, trace_origin = app.state.engine.predict_bytes(bytes(encoded), request_id)
                status = 200
            except PredictionError as error:
                payload, trace, trace_origin, status = {"error": str(error), "request_id": request_id}, error.trace, error.origin_ns, error.status_code
            return payload, trace, trace_origin, status, worker_start

        payload, active_trace, active_origin, status, worker_start = await anyio.to_thread.run_sync(execute)
        serialization_start = perf_counter_ns()
        response = JSONResponse(payload, status_code=status)
        if not app.state.tracing:
            return response
        response_ready = perf_counter_ns()
        stages = [
            {"name": "body_read", "start_ns": 0, "end_ns": body_end - origin},
            {"name": "executor_wait", "start_ns": enqueued - origin, "end_ns": worker_start - origin},
        ]
        for stage in active_trace["stages"]:
            stages.append({"name": stage["name"], "start_ns": active_origin - origin + stage["start_ns"],
                           "end_ns": active_origin - origin + stage["end_ns"]})
        stages.append({"name": "serialize", "start_ns": serialization_start - origin, "end_ns": response_ready - origin})
        server_trace = {"request_id": request_id, "duration_ns": response_ready - origin, "stages": stages}
        response.headers["X-Compact-Trace"] = json.dumps(server_trace, separators=(",", ":"))
        response.headers["X-Request-ID"] = request_id
        return response

    return app


app = create_app()
