"""Explicit commands; service startup never downloads data or trains a model."""

import argparse
from pathlib import Path

from .common import load_config, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.json")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "export", "quantize", "build", "benchmark", "study", "trace"):
        commands.add_parser(name)
    quality = commands.add_parser("evaluate")
    quality.add_argument("--split", choices=["validation", "test"], default="validation")
    serve = commands.add_parser("serve")
    serve.add_argument("--backend", choices=["torch", "fp32", "int8"], default="fp32")
    serve.add_argument("--threads", type=int, default=1)
    serve.add_argument("--port", type=int, default=8000)
    predict = commands.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--backend", choices=["torch", "fp32", "int8"], default="fp32")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare
        prepare(config)
        return
    if args.command == "serve":
        import uvicorn
        from .service import create_app
        uvicorn.run(create_app(args.config, backend=args.backend, threads=args.threads), host="127.0.0.1", port=args.port, workers=1)
        return
    if args.command == "predict":
        import json
        from .runtime import Engine
        result, trace, _ = Engine(config["artifacts"], backend=args.backend).predict_bytes(args.image.read_bytes(), "local-prediction")
        print(json.dumps({"prediction": result, "trace": trace}, indent=2))
        return
    if args.command == "build":
        from .data import prepare
        from .training import train_head
        from .exporting import export_fp32, quantize
        from .evaluation import score
        if (Path(config["artifacts"]) / "checkpoint.pt").exists():
            parser.error("checkpoint already exists; use explicit train/export/quantize commands or choose a new artifacts directory")
        manifest = prepare(config)
        train_head(config, manifest)
        export_fp32(config, manifest)
        try:
            quantize(config, manifest)
        except Exception as error:
            write_json(Path(config["artifacts"]) / "quantization.json", {"status": "unsupported_or_failed", "reason": str(error)})
            print(f"Static INT8 unavailable: {error}. FP32 remains usable.", flush=True)
        score(config, manifest, "validation")
        return
    manifest = read_json(Path(config["artifacts"]) / "splits.json")
    if args.command == "train":
        from .training import train_head
        train_head(config, manifest)
    elif args.command == "export":
        from .exporting import export_fp32
        export_fp32(config, manifest)
    elif args.command == "quantize":
        from .exporting import quantize
        quantize(config, manifest)
    elif args.command == "evaluate":
        from .evaluation import score
        score(config, manifest, args.split)
    elif args.command in ("benchmark", "study"):
        from .experiment import model_study, http_study
        (model_study if args.command == "benchmark" else http_study)(args.config, config, manifest)
    elif args.command == "trace":
        from .experiment import save_trace
        save_trace(config, manifest)


if __name__ == "__main__":
    main()
