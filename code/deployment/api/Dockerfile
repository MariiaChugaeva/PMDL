FROM python:3.10-slim

WORKDIR /app

COPY code/deployment/api/requirements_api.txt /app/requirements_api.txt
RUN pip install --no-cache-dir -r requirements_api.txt

COPY code/deployment/api/main.py /app/main.py
COPY models/digit_recognizer_cnn.h5 /app/models/digit_recognizer_cnn.h5

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
