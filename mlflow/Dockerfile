FROM python:3.12-slim

ARG MLFLOW_VERSION=3.14.0

RUN pip install --no-cache-dir \
    "mlflow[postgres]==${MLFLOW_VERSION}" \
    psycopg2-binary \
    boto3

# Install the CELINE auth plugin (JWT/oauth2-proxy SSO)
COPY packages/celine-mlflow-auth /opt/celine-mlflow-auth
RUN pip install --no-cache-dir /opt/celine-mlflow-auth
COPY packages/celine-mlflow-auth/config/celine_auth.ini /etc/mlflow/celine_auth.ini

EXPOSE 5000
