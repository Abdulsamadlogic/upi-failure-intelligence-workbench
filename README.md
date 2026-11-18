# UPI Failure Intelligence Workbench

[![Open the training notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KNh_n5h8Jq_RjjNHWO7xM9B_T61-Tgiq?usp=sharing)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging%20Face%20Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-Qwen-FFD21E)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-storage-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-evaluation-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

A demonstration workbench for reviewing **simulated UPI transaction failures**. It combines deterministic triage rules with an optional fine-tuned classifier, records transaction event history, and routes model suggestions to a human review queue.


## What it does

- Accepts sanitized transaction events through a FastAPI endpoint.
- Projects event history into a transaction status and evidence summary.
- Applies simple rules first; in hybrid mode, asks a fine-tuned classifier to classify cases the rules cannot resolve.
- Classifies traces into six categories: `network_timeout`, `payer_bank_decline`, `beneficiary_bank_unavailable`, `merchant_confirmation_timeout`, `settlement_mismatch`, or `unknown`.
- Creates review cases for Qwen suggestions and other cases that need attention. A person verifies the evidence and decides what to do.
- Stores events, projections, review decisions, incidents, and audit history in PostgreSQL.
- Provides a small dashboard for reviewing cases and replaying synthetic events.

The Qwen model is a **sequence classifier**: it predicts a category and confidence, not a free-form explanation. The workbench builds evidence and recommended actions from the transaction trace and triage logic.

## Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Model framework | PyTorch |
| Pretrained model and fine-tuning | Hugging Face Transformers, `Qwen/Qwen2.5-0.5B-Instruct` |
| Metrics | scikit-learn |
| API and validation | FastAPI, Pydantic |
| Persistence | PostgreSQL, Psycopg 3 |
| Testing | pytest |
| Notebook training | Google Colab |

## Training notebook

Click the Colab badge at the top of this README to open the training notebook. The notebook uses the pretrained Qwen model and fine-tunes it for the workbench's six-category classification task; it does not train a language model from scratch.

## Data and reported results

The included `data/labeled_incidents.csv` contains **2,000 explicitly synthetic examples**: 1,400 for training, 300 for validation, and 300 for testing. These examples support pipeline development and are not representative of live bank traffic.

Reported results for the existing fine-tuned checkpoint:

| Evaluation | Result | Interpretation |
| --- | ---: | --- |
| Synthetic held-out test split | 100% accuracy (300/300) | Performance on this generated split only; not evidence of real-world accuracy. |
| Rules baseline | 19.7% accuracy; 3% coverage | Baseline on the same synthetic split. |
| Separately worded challenge examples | 11/12 correct | Small, hand-written check; one generic routing-timeout case was classified as beneficiary-bank unavailable, with confidence 1.000. |
| Earlier separately worded examples | 4/6 correct | Small exploratory check. |
| pytest | 9 passed | Previously reported; PostgreSQL integration tests were skipped when no test database URL was configured. |
| API smoke test | HTTP 200; event accepted and review case created | Simulated event only; its suggested category was wrong and went to human review. |

The perfect synthetic score may reflect patterns in the generated data. The challenge set is too small to estimate generalization, and reported confidence is not a guarantee of correctness. Qwen suggestions are routed to human review.

## Run locally

From the repository root:

```powershell
cd upi-codefix-benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The checked-in dataset is already generated. To regenerate it, then fine-tune the model:

```powershell
python -m src.train --generate-synthetic
python -m src.train --epochs 3
```

The default model output directory is `artifacts/qwen-upi-triage`. Evaluate it on the held-out synthetic test split:

```powershell
python -m src.evaluate --model artifacts/qwen-upi-triage
```

The evaluation JSON is written to `artifacts/evaluation.json` by default. To run the tests:

```powershell
pytest -q
```

## Run the API

The API requires a PostgreSQL connection. Set `UPI_DATABASE_URL` in your environment to a database you control before starting it. Keep credentials private; do not commit them or paste them into notebooks or issue reports.

For rules-only mode:

```powershell
$env:UPI_DATABASE_URL = "postgresql://<user>:<password>@<host>:<port>/<database>"
uvicorn src.api:app --reload
```

For hybrid rules-plus-Qwen mode, set the path to a trained checkpoint as well:

```powershell
$env:MODEL_PATH = "artifacts/qwen-upi-triage"
$env:TRIAGE_MODE = "hybrid"
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000` for the dashboard or `http://127.0.0.1:8000/docs` for the interactive API documentation.

Use synthetic events only. Do not enter real account numbers, UPI PINs, OTPs, or customer data.
