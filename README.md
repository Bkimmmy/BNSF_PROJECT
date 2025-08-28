# Predictive Maintenance for Rail Hydraulic/Compressor Systems

## 1. Overview
This project builds a predictive maintenance pipeline that estimates the **Remaining Useful Life (RUL)** of train hydraulic/compressor systems.  
Instead of only classifying *failure vs. no failure*, the system forecasts **how many minutes remain before a failure event**.  
This enables early maintenance scheduling, reduced downtime, and improved reliability.

---

## 2. Data Setup

### Download Dataset
We use the **MetroPT-3 dataset** (real metro compressor/hydraulic IoT signals) hosted by UCI:  
[MetroPT-3 Dataset Link](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset?utm_source=chatgpt.com)

\`\`\`bash
# Download dataset
wget https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip -O metropt3.zip

# Unzip
unzip metropt3.zip -d data/raw/
\`\`\`

### Upload to AWS S3
Replace `<your-bucket>` with your S3 bucket name:

\`\`\`bash
# Upload dataset to S3
aws s3 cp data/raw/ s3://<your-bucket>/metropt3/ --recursive
\`\`\`

---

## 3. Repository Structure
\`\`\`
rail-predictive-maintenance/
├── data/               # Raw & processed datasets (MetroPT-3, weather, census enrichments)
├── notebooks/          # Jupyter notebooks for ETL, modeling, comparison
│   ├── 01_etl_preprocessing.ipynb
│   ├── 02_modeling_rf_xgb_lstm.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_dashboard_prep.ipynb
├── src/                # Python scripts for clean pipeline code
│   ├── etl_pipeline.py
│   ├── train_models.py
│   └── evaluate.py
├── models/             # Trained models (pickle/Torch)
├── dashboards/         # Tableau / Power BI dashboards
├── docker/             # Dockerfile & docker-compose
├── docs/               # Paper, figures, writeups
├── requirements.txt    # Python dependencies
└── README.md
\`\`\`

---

## 4. Data Sources
- **MetroPT-3 Dataset**: IoT signals (pressure, current, temp, GPS).  
- **Weather Data**: NWS/NOAA APIs for environmental enrichment.  
- **Census / GIS Data**: For route and station-level geospatial patterns.  

---

## 5. Models Tested
- Random Forest  
- XGBoost  
- LSTM / GRU (deep learning baselines)  

---

## 6. Results
- **Random Forest** → lowest MAE (~2.3 min)  
- **XGBoost** → best RMSE (~12.2 min), R² (~0.57)  
- **LSTM/GRU** → explored but underperformed (documented in comparison)  

---

## 7. How to Run
\`\`\`bash
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
\`\`\`

---

## 8. Deployment
Containerized with Docker:

\`\`\`bash
docker build -t rail_maintenance .
docker run rail_maintenance
\`\`\`

---

## 9. Deliverables
- ETL Pipeline (PySpark + Python)  
- Modeling Notebooks (RF, XGB, LSTM, GRU)  
- Comparison Report (PDF in \`/docs/\`)  
- Dashboards (Tableau Public, Power BI Desktop)  
- Docker Container (deploy-ready)  
- Loom Walkthrough (linked in docs & README)  
"""
