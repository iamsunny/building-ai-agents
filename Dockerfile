# Part 11: container for the helpdesk service.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build the handbook index before building the image (python -m part05_rag.ingest),
# so data/handbook_index.json is copied in and the container needs no embedding call at start.

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# One worker while the checkpointer is in memory. With a Postgres checkpointer you can run more
# workers or replicas, because every one of them reads the same conversation state.
CMD ["uvicorn", "part11_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
