FROM python:3.11-slim

# 1) System deps (optional: libgomp for xgboost; here, scikit-learn only)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# 2) Workdir and copy only requirements first (better layer caching)
WORKDIR /app
COPY requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code, data (optional), and trained models
COPY src /app/src
COPY api /app/api
COPY models /app/models

# 4) Expose FastAPI
EXPOSE 8000

# 5) Default command: serve the API
CMD ["streamlit", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]