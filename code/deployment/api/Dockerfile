FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code ./code
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
