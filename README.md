# UPI Failure Intelligence Workbench

[![Open the training notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KNh_n5h8Jq_RjjNHWO7xM9B_T61-Tgiq?usp=sharing)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging%20Face%20Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Qwen-FFD21E)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-storage-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-evaluation-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end machine-learning workbench for classifying simulated UPI transaction failures and supporting human-reviewed triage.

The project fine-tunes the pretrained **Qwen2.5-0.5B-Instruct** model as a six-class sequence classifier, then integrates it with a rules-first workflow, a FastAPI service, PostgreSQL event history, and a review dashboard.

## Highlights

- Fine-tunes a pretrained Qwen model to classify UPI failure traces.
- Combines deterministic rules with model predictions for unresolved cases.
- Builds transaction status and evidence summaries from event history.
- Routes cases for human review and records review and audit history.
- Includes a dashboard for reviewing cases and replaying synthetic events.
- Provides training, evaluation, and test workflows.

## How it works

```text
Transaction events
        ↓
Status projection and evidence summary
        ↓
Rules-first triage
        ↓
Fine-tuned Qwen classifier for unresolved cases
        ↓
Human review queue
        ↓
PostgreSQL event, decision, and audit history
```

The classifier predicts a failure category and confidence score. The workbench derives evidence and recommended actions from the transaction trace and triage logic.

## Failure categories

The classifier recognizes six categories:

- `network_timeout`
- `payer_bank_decline`
- `beneficiary_bank_unavailable`
- `merchant_confirmation_timeout`
- `settlement_mismatch`
- `unknown`

## Model and dataset

The classifier is fine-tuned from `Qwen/Qwen2.5-0.5B-Instruct` using Hugging Face Transformers and PyTorch.

The included dataset contains **2,000 synthetic examples**:

| Split | Examples |
| --- | ---: |
| Training | 1,400 |
| Validation | 300 |
| Test | 300 |

## Evaluation

Latest notebook results:

| Evaluation | Result |
| --- | ---: |
| Fine-tuned model accuracy on 300-example held-out synthetic set | 100% |
| Fine-tuned model macro F1 on held-out synthetic set | 100% |
| Hand-written challenge cases | 11/12 correct |
| Automated tests | 9 passed, 4 skipped |

The hand-written challenge set provides an additional check using separately worded examples. Reported dataset scores are for the synthetic test set.

## Technology stack

- **Model:** Qwen2.5-0.5B-Instruct, PyTorch, Hugging Face Transformers
- **API and validation:** FastAPI, Pydantic
- **Data and evaluation:** Python, scikit-learn
- **Persistence:** PostgreSQL, Psycopg 3
- **Testing:** pytest
- **Training environment:** Google Colab

## Run locally

Clone the repository and enter the benchmark directory:

```powershell
git clone https://github.com/samad-node/upi-failure-intelligence-workbench.git
cd upi-failure-intelligence-workbench/upi-codefix-benchmark

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The checked-in dataset is ready to use. To fine-tune the model with the notebook's configuration:

```powershell
python -m src.train --dataset data/labeled_incidents.csv --output-dir artifacts/qwen-upi-triage --epochs 1 --batch-size 2
```

Evaluate the saved model:

```powershell
python -m src.evaluate --dataset data/labeled_incidents.csv --model artifacts/qwen-upi-triage
```

Run the test suite:

```powershell
python -m pytest -q
```

Four tests may be skipped when a PostgreSQL test database is not configured.

## Run the API

The API requires a PostgreSQL connection. Set `UPI_DATABASE_URL` to a database you control.

For rules-only mode:

```powershell
$env:UPI_DATABASE_URL = "postgresql://<user>:<password>@<host>:<port>/<database>"
uvicorn src.api:app --reload
```

For hybrid rules-plus-Qwen mode:

```powershell
$env:MODEL_PATH = "artifacts/qwen-upi-triage"
$env:TRIAGE_MODE = "hybrid"
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000` for the dashboard or `http://127.0.0.1:8000/docs` for the API documentation.
