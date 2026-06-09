FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow[postgres]>=2.10.0" \
    psycopg2-binary \
    boto3

EXPOSE 5000
