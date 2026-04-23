---
title: RevenueShield Telecom AI
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 📡 Telecom Customer Churn AI Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat-square&logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=flat-square)

**Real-time customer churn risk scoring — serving Random Forest via FastAPI on Hugging Face Spaces.**

[Live Demo](https://nilotpaldhar2004.github.io/Telecom-AI-Predictor/) · [AI Engine (HF)](https://huggingface.co/spaces/nilotpaldhar2004/telecom-churn-ai) · [Report a Bug](https://github.com/nilotpaldhar2004/Telecom-AI-Predictor/issues)

</div>

---

## 📸 What it does

Submit 19 customer attributes — demographics, service subscriptions, billing details — and the model returns an instant **STAY / CHURN** decision with:
- Churn probability as a percentage
- A **High / Medium / Low** risk classification
- An animated risk gauge and dual probability bars
- Server-side inference latency in milliseconds

---

## ✨ Features

- **Real-Time Inference** — Sub-15ms predictions via a FastAPI + Uvicorn backend
- **Animated Risk Gauge** — SVG arc gauge fills dynamically with color-coded risk levels
- **19-Feature Model** — Covers demographics, contracts, internet services, billing, and add-ons
- **Pydantic Validation** — All 19 input fields are strictly type-checked before reaching the model
- **Request Log** — Live terminal-style log panel shows every request, RTT, and risk level
- **Keep-Alive Endpoint** — `/ping` prevents Render free-tier cold starts when monitored by UptimeRobot
- **Full API Docs** — Auto-generated Swagger UI at `/docs` and ReDoc at `/redoc`
- **Responsive UI** — Works on mobile, tablet, and desktop

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python, Pandas, NumPy, Scikit-Learn, Random Forest |
| Hyperparameter Tuning | GridSearchCV / RandomizedSearchCV |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Render (API) + GitHub Pages (Frontend) |

---

## 📂 Project Structure

```
Telecom-AI-Predictor/
│
├── main.py                                         # FastAPI backend server
├── index.html                                      # Frontend dashboard
├── End-to-End ML Pipeline for Customer Churn Predict.ipynb  # Training notebook
├── requirements.txt                                # Python dependencies
├── LICENSE.txt                                     # MIT License
├── .gitignore                                      # Git ignore rules
├── README.md                                       # Project documentation
│
└── model.pkl                                       # (gitignored — not committed)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher

### 1. Clone the repository

```bash
git clone https://github.com/nilotpaldhar2004/Telecom-AI-Predictor.git
cd Telecom-AI-Predictor
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the trained model

Place `model.pkl` in the project root (same folder as `main.py`).

> **Don't have the model?** Run the Jupyter notebook `End-to-End ML Pipeline for Customer Churn Predict.ipynb` from start to finish — it trains and saves `model.pkl` automatically.

### 4. Start the server

```bash
python main.py
```

Open `http://localhost:8000` in your browser — FastAPI serves `index.html` directly.

---

## 🌐 Deployment

| Component | Host | URL |
|---|---|---|
| Frontend (`index.html`) | GitHub Pages | `https://nilotpaldhar2004.github.io/Telecom-AI-Predictor/` |
| Backend (`main.py`) | Render | `https://telecom-ai-predictor.onrender.com` |

### Deploy to Render

1. Push your code to GitHub (`model.pkl` is gitignored — add it via Render Disk or environment)
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment:** Python 3
5. Deploy — Render gives you a public URL

> **Tip:** Create a free UptimeRobot monitor pointing at `https://your-app.onrender.com/ping` every 10 minutes to keep the server warm and avoid cold-start delays.

### Deploy frontend to GitHub Pages

1. **Settings → Pages → Source → main branch → / (root)**
2. Save — GitHub Pages serves `index.html` automatically within ~60 seconds

---

## 📡 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "2.0.0"
}
```

### `GET /ping`

```json
{ "pong": true }
```

### `POST /predict`

**Request body (19 fields):**

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.20
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 74.81,
  "risk_level": "High",
  "result": "Customer Will Churn",
  "latency_ms": 8.23
}
```

Full interactive documentation is available at `/docs` (Swagger UI) when the server is running.

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Dataset | IBM Telco Customer Churn |
| Input Features | 19 |
| Risk Thresholds | High > 70% · Medium 30–70% · Low < 30% |
| Output | Binary (0 = Stay, 1 = Churn) + probability |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.

---

## 👤 Author

**Nilotpal Dhar** · [@nilotpaldhar2004](https://github.com/nilotpaldhar2004) · March 2026

---

<div align="center">
  <sub>Built with Python, FastAPI, and Scikit-Learn · Deployed on Render + GitHub Pages</sub>
</div>