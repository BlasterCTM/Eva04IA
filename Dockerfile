FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=models/modelo_demanda_v1.joblib \
    LOG_LEVEL=info

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY . /app

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn src.serving.api_endpoints_only:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]
