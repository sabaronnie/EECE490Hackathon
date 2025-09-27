# Use slim Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what’s needed for runtime
COPY src/ /app/src/
COPY models/ /app/models/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "src/website/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
