FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN echo "Simulating model download for Run ID: ${RUN_ID}"

CMD ["sh", "-c", "echo Downloaded model for Run ID: ${RUN_ID}"]