FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/cyber_ai_model.pkl \
    PORT=8001

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_ai.py ai_guard_service.py cyber_ai_model.pkl ./

EXPOSE 8001

CMD ["sh", "-c", "uvicorn ai_guard_service:app --host 0.0.0.0 --port ${PORT:-8001}"]
