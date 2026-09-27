# syntax=docker/dockerfile:1
# mock-dcgm-exporter: DCGM-SHAPED telemetry simulator. Not a GPU; every
# series carries simulation="true" (see deploy/k8s/README.md scope note).
FROM python:3.12-slim
COPY mock_dcgm.py /app/mock_dcgm.py
ENV PORT=9400 \
    PYTHONUNBUFFERED=1
EXPOSE 9400
CMD ["python3", "/app/mock_dcgm.py"]
