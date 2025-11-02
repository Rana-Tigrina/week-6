# Start from a base Python image (lightweight version)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY iris_fast_api.py .

# Copy your trained model
COPY model.joblib .

# Tell Docker this container listens on port 8200
EXPOSE 8200

# Command to run when container starts
CMD ["uvicorn", "iris_fast_api:app", "--host", "0.0.0.0", "--port", "8200"]
