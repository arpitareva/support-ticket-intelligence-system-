# Single image running both the API (8000) and the Streamlit UI (8501).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/support_tickets.csv ./data/support_tickets.csv

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -fs http://localhost:8000/health || exit 1

# The API must be up before the UI's first render; the UI only reads the API.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
     streamlit run frontend/streamlit_app.py \
       --server.port 8501 --server.address 0.0.0.0"]
