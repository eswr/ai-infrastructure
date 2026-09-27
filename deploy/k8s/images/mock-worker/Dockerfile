# syntax=docker/dockerfile:1
# Mock worker image: the zero-dependency Python failure-injection server,
# plus curl so `kubectl exec` scenarios can drive /admin/knob and /drain.
FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY mock_worker.py /app/mock_worker.py
# Containers must bind the non-loopback interface to be a k8s endpoint.
ENV HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python3", "/app/mock_worker.py"]
