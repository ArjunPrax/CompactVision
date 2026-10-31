# Attribution and provenance

## Foundations

- [ONNX Runtime](https://github.com/microsoft/onnxruntime): runtime and quantization APIs, MIT-licensed upstream code.
- [PyTorch](https://github.com/pytorch/pytorch) and [TorchVision](https://github.com/pytorch/vision): neural-network computation, MobileNetV3-Small, ImageNet weights and CIFAR-10 loader. TorchVision identifies BSD-3-Clause licensing; weight/data terms are separate. Exact versions are in `uv.lock`; training metadata records the weight URL/digest.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky, Vinod Nair and Geoffrey Hinton. Cite Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. TorchVision checks the official archive MD5. Dataset images are not committed/released.
- FastAPI, Uvicorn, Pillow, NumPy and HTTPX provide serving, imaging and client infrastructure; the lock file identifies dependencies.

MobileNetV3 originates in Howard et al., *Searching for MobileNetV3*, ICCV 2019. The architecture, pretrained weights and runtime/quantization algorithms come from these upstream sources.

## This implementation

My contribution is the application and evaluation code in `compact_vision/` and `benchmarks/`: shared preprocessing checks, frozen-feature head training, ONNX export/parity, a static INT8 experiment, CPU serving, stage timings, request-outcome accounting and a controlled threading comparison.

The measurements in this repository identify the source revision, model, inputs, configuration and hardware used. Synthetic fixtures test implementation contracts and are excluded from model-quality and performance claims.

Training the last linear layer over frozen ImageNet features is transfer learning with a frozen backbone. The evaluated task is CIFAR-10 classification. Dataset files and generated weights are excluded from Git; the build command retrieves public source data and pretrained weights under their respective terms. This project uses public sources and no employer, lab or private research assets.
