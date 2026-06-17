# Run AG News Microservice Locally in VS Code

## 1. Open this folder in VS Code

Open the `agnews_microservice` folder.

## 2. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Train and save the model

Quick test run:

```powershell
python train_agnews_tfidf.py --train-per-class 500 --test-per-class 100
```

Recommended full-data training:

```powershell
python train_agnews_tfidf.py --full-data --max-features 100000 --min-df 1 --c-value 4.0
```

Optional hyperparameter tuning run:

```powershell
python train_agnews_tfidf.py --full-data --tune
```

This creates:

```text
outputs/agnews_tfidf_logreg.joblib
outputs/agnews_metrics.json
outputs/sample_inputs.json
```

## 5. Start the API service

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 6. Open the demo page

```text
http://127.0.0.1:8000/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 7. Test the service from PowerShell

Open a second PowerShell window in the same folder and run:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"text":"Apple announced a new AI feature for its latest iPhone."}'
```

Expected output format:

```json
{
  "predicted_category": "Sci_Tech",
  "confidence": 0.84,
  "probabilities": {
    "Business": 0.05,
    "Sci_Tech": 0.84,
    "Sports": 0.01,
    "World": 0.10
  }
}
```
