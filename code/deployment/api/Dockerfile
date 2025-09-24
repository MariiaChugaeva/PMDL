FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY code/deployment/api/requirements_api.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire code structure
COPY code /app/code

EXPOSE 8000

# Run from the correct directory context
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]