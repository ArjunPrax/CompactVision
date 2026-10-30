FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.162.0 uvicorn==0.54.0 onnxruntime==1.31.1 numpy==2.5.4 pillow==12.3.0
COPY compact_vision/ compact_vision/
COPY benchmarks/ benchmarks/
COPY configs/demo.json configs/demo.json
ENV COMPACT_VISION_ARTIFACTS=/models
EXPOSE 8000
CMD ["uvicorn", "compact_vision.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
