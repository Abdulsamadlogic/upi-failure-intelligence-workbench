FROM python:3.11-slim

WORKDIR /app

COPY upi-codefix-benchmark/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY upi-codefix-benchmark/src ./src
COPY upi-codefix-benchmark/data ./data

ENV TRIAGE_MODE=rules
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
