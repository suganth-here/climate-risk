# Climate Risk-Based Credit Scoring & Lending Intelligence System

Full-stack split with:
- Frontend: React (`frontend/`)
- Backend: Django API (`backend/`)
- Core ML/climate logic and trained artifacts: unchanged in `src/`, `dataset/`, `artifacts/`

## Architecture
- `backend/api/logic.py` reuses the same climate/loan decision flow previously in `app.py`
- `src/climate_intelligence.py`, `src/data_loader.py`, and trained model artifacts are preserved as-is
- `frontend/src/App.jsx` provides the same content sections:
  - Loan Inputs
  - Annual Risk Points
  - 50-Year Projection and Tenure Risk Graph
  - Portfolio Risk Analysis (CSV Upload)

## Backend setup (Django)
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python backend/manage.py runserver 127.0.0.1:8000
```

## Frontend setup (React)
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://127.0.0.1:5173` and proxies `/api/*` to Django `http://127.0.0.1:8000`.

## One-click launcher (Windows)
- `start_fullstack.bat`
- or `.\start_fullstack.ps1`

## API endpoints
- `GET /api/metadata/`
- `POST /api/predict/`
- `POST /api/portfolio/analyze/`

## Portfolio CSV format
- `property_id`
- `latitude`
- `longitude`
- `loan_amount`
- `property_value`
- `tenure_years`
- `asset_type`

Sample file: `data/sample_properties.csv`
