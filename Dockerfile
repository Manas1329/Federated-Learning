FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY federated_healthcare/ ./federated_healthcare/
# Data can be mounted at runtime, but we copy it here for standalone usage if needed
# COPY data/ ./data

ENV PYTHONPATH=/app

CMD ["python", "federated_healthcare/src/client.py"]