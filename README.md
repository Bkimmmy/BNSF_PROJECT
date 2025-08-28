# Predictive Maintenance for Rail Hydraulic/Compressor Systems

## Overview
This project builds a predictive maintenance pipeline that estimates the Remaining Useful Life (RUL) of train hydraulic/compressor systems.  
Instead of only classifying "failure vs. no failure," the system forecasts how many minutes remain before a failure event.  
This enables early maintenance scheduling, reduced downtime, and improved system reliability.

## Repository Structure
rail-predictive-maintenance/
├── data/ # Raw & processed datasets (MetroPT-3, weather, census enrichments)
├── notebooks/ # Jupyter notebooks for ETL, modeling, comparison
│ ├── 01_etl_preprocessing.ipynb
│ ├── 02_modeling_rf_xgb_lstm.ipynb
│ ├── 03_model_comparison.ipynb
│ └── 04_dashboard_prep.ipynb
├── src/ # Python scripts for clean pipeline code
│ ├── etl_pipeline.py
│ ├── train_models.py
│ └── evaluate.py
├── models/ # Trained models (pickle/Torch)
├── dashboards/ # Tableau / Power BI dashboards
├── docker/ # Dockerfile & compose for containerization
├── docs/ # Paper/PDF writeup, math appendix, figures
├── requirements.txt # Python dependencies
└── README.md # Project overview

bash
Copy code

## Data Sources
- **MetroPT-3 Dataset**: Real metro compressor/hydraulic IoT signals (pressure, current, temp, GPS).  
- **Weather Data**: Enriched via NWS/NOAA APIs.  
- **Census / GIS Data**: Geospatial enrichment for routes and station-level patterns.  

## Models Tested
- Random Forest  
- XGBoost  
- LSTM / GRU (deep learning baselines)

## Results
- Random Forest → lowest MAE (~2.3 min)  
- XGBoost → best RMSE (~12.2 min) and R² (~0.57)  
- LSTM/GRU → explored but underperformed (documented in comparison)  

## How to Run
```bash
# Clone repo
git clone <your_repo_url>
cd rail-predictive-maintenance

# Create environment & install deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run ETL
python src/etl_pipeline.py

# Train models
python src/train_models.py

# Evaluate
python src/evaluate.py
Deployment
Dockerized pipeline:

bash
Copy code
docker build -t rail_maintenance .
docker run rail_maintenance
Deliverables
ETL Pipeline (PySpark + Python)

Modeling Notebooks (RF, XGB, LSTM, GRU)

Comparison Report (math + metrics, PDF in /docs/)

Dashboards (Tableau Public, Power BI Desktop)

Docker Container (deploy-ready)

Loom Walkthrough (linked in docs & README)
