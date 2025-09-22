FROM python:3.10-slim

WORKDIR /app

COPY code/deployment/app/requirements_app.txt /app/requirements_app.txt
RUN pip install --no-cache-dir -r requirements_app.txt

COPY code/deployment/app/app.py /app/app.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
