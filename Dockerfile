# Use slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy, pandas, scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit’s default port
EXPOSE 8501

# Launch Streamlit when the container starts
CMD ["streamlit", "run", "src/website/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
