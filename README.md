# AdEva Final ⚡ — Energy‑Load Forecasting & MLOps Pipeline

> Transformer‑based forecasting with automated CI + CD, experiment tracking, and cloud‑agnostic deployment — built for the **Ausgrid** time‑series dataset and Windows‑friendly tooling.

---

## 🔑 Key Features

| Module | What it does | Location |
|--------|--------------|----------|
| **Data Ingestion & Transformation** | Downloads raw CSVs, cleans, normalises, and slides a fixed window over the series to create `(X, y)` pairs. | `src/AdEva_final/pipeline/data_ingestion.py` & `data_transformation.py` |
| **Time‑Series Transformer** | Lightweight encoder (multi‑head self‑attention) with sinusoidal positional encodings. Outputs a 7‑day forecast. | `pipeline/model_trainer.py` |
| **Dual‑Format Model Export** | After training, weights are saved as<br>• `model.pt` (PyTorch `state_dict`)<br>• `model.h5` (true HDF‑5 — Keras‑/TF‑compatible) | `models/` |
| **Experiment Tracking** | Hyper‑params & metrics (loss/epoch) logged to MLflow **remote or local** backend. | `mlflow/` |
| **Continuous Integration** | `ci.yml` (GitHub Actions) creates the Conda env, runs unit tests, trains on a small subset, uploads `transformer-model` artifact. | `.github/workflows/ci.yml` |
| **Continuous Deployment** | `cd.yml` downloads the artifact after every green CI run and pushes the fresh `transformer.h5` to any of —<br>1. another GitHub repo (PAT)<br>2. SMB / IIS folder<br>3. S3 / MinIO bucket | `.github/workflows/cd.yml` |

---

## 🔧 Prerequisites

* **Python ≥3.10**
* **Conda 4.12+** (or Mamba for speed)
* GPU optional (CUDA 11.x); falls back to CPU
* MLflow server (any backend store; SQLite works for dev)

---

## 🚀 Quick Start

```bash
# 1 — clone
$ git clone https://github.com/nutsfinder/AdEva_final.git && cd AdEva_final

# 2 — create env & install deps
$ conda env create -n adeva_env -f environment.yml   # OR:
$ conda create -n adeva_env python=3.10 && conda activate adeva_env
$ pip install -r requirements.txt

# 3 — pre‑process the dataset (one‑off)
$ python src/AdEva_final/pipeline/data_ingestion.py
$ python src/AdEva_final/pipeline/data_transformation.py

# 4 — train & log to MLflow
$ python src/AdEva_final/pipeline/model_trainer.py
```

Open **MLflow UI**: `mlflow ui -h 0.0.0.0 -p 5000` → http://localhost:5000

---

## 🛠 Project Structure

```
AdEva_final/
├── .github/workflows/    # CI & CD pipelines
├── models/              # *.pt & *.h5 artefacts (git‑ignored)
├── notebooks/           # exploratory notebooks
├── src/AdEva_final/
│   ├── config/          # centralised YAML/JSON configs
│   └── pipeline/
│       ├── data_ingestion.py
│       ├── data_transformation.py
│       └── model_trainer.py
└── requirements.txt
```

---

## 🤖 CI & CD in One Minute

1. **Commit / PR open** → **CI** spins up ➟ trains ➟ uploads `transformer-model` artifact.
2. CI finishes green ➟ **CD** auto‑fires (`workflow_run`) ➟ downloads artifact.
3. CD pushes `transformer.h5` to the **destination** you configured (another repo, local share, S3…).

Secrets used:

* `APP_PAT`  — Personal‑access token (repo‑scope) for cross‑repo pushes  *(or use `GITHUB_TOKEN` for same‑repo writes).*  
* Any cloud credentials (AWS keys, SMB user…) stored the same way.

---

## 📦 Consuming the Model

```python
import torch, h5py
from AdEva_final.pipeline.model_trainer import TimeSeriesTransformer

with h5py.File("transformer.h5") as f:
    state = {k: torch.tensor(f[k][:]) for k in f}

model = TimeSeriesTransformer(feature_size=FEATURES)
model.load_state_dict(state)
model.eval()
forecast = model(input_window_tensor)
```

---

## 🧪 Testing & Smoke Checks

* **Unit tests**: `pytest -q tests/` (covers windowing logic and forward pass)
* **GitHub Actions**: green badge → every step succeeded, artefacts uploaded, model deployed.
* **Manual**: pull deployed `.h5`, run the snippet above, confirm `(batch, 7)` output.

---

## 🙏 Contributing

PRs & issues welcome! Please:

1. Fork ➟ feature branch ➟ add tests.
2. Run `black` & `ruff`.
3. Verify `pytest` + local MLflow run pass.
4. Open PR; GitHub Actions will comment with CI + CD results.

---

## 📝 License

© 2025 — NutsFinder. Released under the MIT License — see [`LICENSE`](LICENSE).

