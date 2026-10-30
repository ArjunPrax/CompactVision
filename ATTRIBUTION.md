# Attribution and provenance

## Foundations

- [ONNX Runtime](https://github.com/microsoft/onnxruntime): runtime and quantization APIs, MIT-licensed upstream code.
- [PyTorch](https://github.com/pytorch/pytorch) and [TorchVision](https://github.com/pytorch/vision): neural-network computation, MobileNetV3-Small, ImageNet weights and CIFAR-10 loader. TorchVision identifies BSD-3-Clause licensing; weight/data terms are separate. Exact versions are in `uv.lock`; training metadata records the weight URL/digest.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky, Vinod Nair and Geoffrey Hinton. Cite Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. TorchVision checks the official archive MD5. Dataset images are not committed/released.
- FastAPI, Uvicorn, Pillow, NumPy and HTTPX provide serving, imaging and client infrastructure; the lock file identifies dependencies.

MobileNetV3 originates in Howard et al., *Searching for MobileNetV3*, ICCV 2019. Architecture, pretrained ImageNet results and runtime/quantization algorithms are not personal inventions.

## This implementation

The independently written code in `compact_vision/` and `benchmarks/` integrates shared checked preprocessing, frozen-feature head training, export/parity, a static INT8 experiment, CPU serving, stage/outcome reporting and a controlled threading decision. No code was copied from the unavailable earlier personal project or employer/lab assets.

The owner authorized a fresh rebuild on 30 October 2026. Earlier approximate size/speed/accuracy reports stay historical user reports; they are not measurements for this build. New claims require this code/model/input/hardware provenance.

Training a new last linear layer over frozen ImageNet features is transfer learning/head training, not full-backbone fine-tuning. CIFAR-10 is a public-image demo, not workshop-parts recognition or production operation. Synthetic fixtures/fake engines test contracts only. Data and generated weights stay ignored; no release is authorized.
