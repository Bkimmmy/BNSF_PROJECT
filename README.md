# Rail Predictive Maintenance
**Forecasting Remaining Useful Life (RUL) of train compressor/hydraulic systems**

---

## 1. Overview
This project develops a predictive maintenance pipeline for train hydraulic/compressor systems.  
Instead of a binary failure vs. no-failure model, it estimates the Remaining Useful Life (RUL) in minutes.

This enables operators to:
- Anticipate failures before they occur
- Optimize maintenance schedules
- Reduce downtime and costs
- Improve reliability and safety of rolling stock

We benchmark classical ML (Random Forest, XGBoost) against deep learning (LSTM, GRU) to answer:
> Does deep learning add value for predictive maintenance in real-world metro sensor data?

---

## 2. Data Sources

### MetroPT-3 Dataset (UCI Repository)

[MetroPT-3 Dataset (UCI Repository)](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset?utm_source=chatgpt.com)


Real-world IoT logs from metro train compressors, including:
- Motor current
- Oil temperature
- DV pressure
- Digital valve signals (COMP, MPG, LPS)
- GPS location
- Failure labels

### Weather Data (NOAA/NWS APIs)
Matched by GPS and timestamp to capture environmental effects (temperature, humidity, pressure).

### Geospatial Data (Census TIGER / OpenStreetMap) *(optional)*
Maps each GPS point to rail segments to check if route-level factors influence failures.

---

## 3. Repository Structure
```text
rail-predictive-maintenance/
├── data/                  # Raw and processed datasets
│   ├── raw/               # MetroPT-3 + enrichments
│   └── processed/         # Features ready for modeling
├── notebooks/             # Analysis and experiments
│   ├── 01_etl_preprocessing.ipynb
│   ├── 02_modeling_rf_xgb_lstm.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_deep_sequence_models.ipynb
│   └── 05_model_serving_and_eval.ipynb
├── src/                   # Production-ready Python modules
│   ├── etl_pipeline.py
│   ├── train_models.py
│   └── evaluate.py
├── models/                # Trained models (Pickle/Torch)
├── dashboards/            # Tableau / Power BI dashboards
├── docker/                # Dockerfile + compose
├── docs/                  # PDF writeup, figures
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 4. Pipeline

### ETL & Feature Engineering (`src/etl_pipeline.py`)
- Read raw IoT logs from AWS S3  
- Clean missing values, normalize units  
- Create rolling aggregates (mean, RMS, variance)  
- Label failures (COMP=0, MPG=0, LPS=0)  
- Compute Remaining Useful Life (RUL)

### Model Training (`src/train_models.py`)
- Random Forest (baseline)  
- XGBoost (gradient boosting)  
- LSTM & GRU (deep sequence models with PyTorch)  
- Hyperparameter tuning

### Evaluation (`src/evaluate.py`)
- Metrics: MAE, RMSE, R², sMAPE  
- Error distribution plots  
- Residual analysis  
- Comparison across classical vs deep learning

### Visualization (`dashboards/`)
- Tableau & Power BI dashboards with KPIs  
- Interactive filtering by route, weather, station

### Deployment (`docker/`)
- Dockerized ETL + training pipeline  
- Deployable inference API (Flask/AWS Lambda ready)

---

## 5. Models Tested

**Random Forest**  
- Simple, interpretable, strong baseline

**XGBoost**  
- Best balance of accuracy and generalization

**LSTM / GRU**  
- Sequence models to leverage temporal context  
- Tested with different hidden sizes, layers, weighted loss

---

## 6. Results

| Model         | MAE (min) | RMSE (min) | R²  | ≤5 min % | ≤10 min % |
|---------------|-----------:|-----------:|:---:|---------:|----------:|
| Random Forest | ~2.3       | ~13.5      | 0.54 | 93%      | 98%       |
| XGBoost       | ~2.5       | 12.2       | 0.57 | 91%      | 97%       |
| LSTM          | ~5–6       | 19–21      | <0  | 70–75%   | 85–88%    |
| GRU           | ~5–6       | 20–22      | <0  | 68–72%   | 83–86%    |

**Key insight:** Tree-based models (Random Forest, XGBoost) outperformed deep sequence models on this dataset.  
Feature-engineered signals capture most predictive power; deep learning was limited by dataset size and noise.

---

## 7. How to Run
```bash
# Clone repo
git clone https://github.com/<your-username>/rail-predictive-maintenance.git
cd rail-predictive-maintenance

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run ETL
python src/etl_pipeline.py

# Train models
python src/train_models.py

# Evaluate
python src/evaluate.py
```

---

## 8. Deployment

### Run with Docker
```bash
# Build image
docker build -t rail_maintenance .

# Run container
docker run rail_maintenance
```

### AWS (Optional)
- Store raw and processed data in S3  
- Deploy models via AWS Lambda + API Gateway  
- Monitor predictions with CloudWatch

---

## 9. Deliverables
- ✅ ETL Pipeline (PySpark + AWS S3 integration)  
- ✅ Model Training (RF, XGB, LSTM, GRU)  
- ✅ Evaluation Framework (metrics, plots, bias checks)  
- ✅ Dashboards (Tableau Public, Power BI Desktop)  
- ✅ Dockerized Pipeline (ready for deployment)  
- ✅ Paper-Style Report (`/docs/`)  
- ✅ Loom Walkthrough (linked in README and docs)

---

## 10. Next Steps
- Integrate real-time inference with streaming IoT data  
- Explore anomaly detection via autoencoders  
- Expand to GIS-based risk mapping across rail networks  
- Implement model monitoring & drift detection in production
