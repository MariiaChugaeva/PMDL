# code/deployment/app/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_app.txt /app/requirements_app.txt
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/requirements_app.txt

# Copy app code
COPY app.py /app/app.py

EXPOSE 8501

# Streamlit config to avoid opening a browser
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
