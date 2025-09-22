# code/deployment/api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps for scikit-learn slightly faster installs
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt /app/requirements_api.txt
RUN pip install --no-cache-dir -r /app/requirements_api.txt

# Copy API code
COPY main.py /app/main.py

# Copy model files (we expect you to have models in repo root ./models)
COPY ../../models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
