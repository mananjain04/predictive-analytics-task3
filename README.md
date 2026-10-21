# Predictive Analytics Using Historical Data
## Power BI & Data Analytics Internship — Task 3

![Task Status](https://img.shields.io/badge/Task-Completed-success)
![Python Version](https://img.shields.io/badge/Python-3.14-blue)
![Power BI Ready](https://img.shields.io/badge/Power_BI-Compatible-yellow)

---

## 📌 Project Overview
This repository contains the end-to-end solution for **Task 3: Predictive Analytics Using Historical Data**.

The objective is to **build predictive models to forecast future trends**, clean and preprocess historical datasets, evaluate model accuracy, visualize predictions, and structure the outputs for interactive dashboarding in **Power BI**.

---

## 📂 Repository Structure

```
├── data/
│   ├── raw_historical_sales_data.csv          # Raw 5-year operational dataset (with anomalies)
│   ├── cleaned_historical_sales_data.csv      # Cleaned, preprocessed & engineered dataset
│   ├── model_evaluation_metrics.csv           # Accuracy benchmarks (MAE, RMSE, MAPE, R2)
│   ├── future_forecast_predictions.csv        # 90-day forward predictions (Q1 2026)
│   ├── power_bi_master_sales_forecast.csv     # Unified Actuals + Forecast table for Power BI
│   └── Sales_Predictive_Analytics_Data.xlsx   # All-in-one multi-tab Excel workbook
├── visualizations/
│   ├── 01_eda_trend_seasonality.png           # 5-Year Trend, Monthly & Weekly seasonality
│   ├── 02_regression_predictions.png          # Actual vs Predicted & Feature Importances
│   ├── 03_time_series_forecasting.png         # Historical trajectory + 90-day forecast & 95% CI
│   └── 04_model_accuracy_comparison.png       # Comparative model error & R2 charts
├── generate_dataset.py                        # Synthetic data generation script
├── predictive_modeling.py                     # Cleaning, ML training, evaluation, & forecasting
├── power_bi_python_visual.py                  # Copy-paste script for Power BI Python visual
├── Power_BI_Dashboard_Guide.md                # Step-by-step Power BI tutorial & DAX measures
├── Task3_Predictive_Analytics_Report.md       # Comprehensive technical project report
└── README.md                                  # Repository overview and quick start guide
```

---

## ⚙️ How to Run the Pipeline

### 1. Requirements
Ensure required Python packages are installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels xlsxwriter
```

### 2. Generate Raw Dataset
```bash
python generate_dataset.py
```

### 3. Run Preprocessing, Machine Learning & Forecasting
```bash
python predictive_modeling.py
```
This script will:
- Deduplicate, impute missing values, and cap statistical outliers.
- Engineer time, lag, and rolling window features.
- Fit **5 predictive models**:
  1. Linear Regression
  2. Ridge Regularized Regression
  3. Random Forest Regressor
  4. Holt-Winters Triple Exponential Smoothing
  5. SARIMA
- Evaluate model accuracy (MAE, RMSE, MAPE %, $R^2$).
- Generate 90-day forward predictions for Q1 2026 with 95% Confidence Intervals.
- Export all CSVs, the Excel workbook, and high-resolution chart images.

---

## 📊 Model Accuracy Evaluation Summary

| Model | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **$1,353.44** | **$1,876.23** | **2.60%** | **0.8177** | **Optimal Model** |
| **Ridge Regression** | $1,356.09 | $1,890.27 | 2.61% | 0.8149 | Top Performer |
| **Random Forest** | $2,067.47 | $2,776.55 | 3.82% | 0.6007 | Non-Linear Fit |
| **Holt-Winters ES** | $4,049.22 | $4,497.75 | 7.79% | -0.0477 | Univariate Baseline |
| **SARIMA** | $4,737.05 | $5,278.88 | 9.24% | -0.4433 | Univariate Baseline |

---

## 📈 Power BI Integration

Read [`Power_BI_Dashboard_Guide.md`](Power_BI_Dashboard_Guide.md) for complete instructions:
1. Load `data/power_bi_master_sales_forecast.csv` or `data/cleaned_historical_sales_data.csv`.
2. Add recommended DAX measures for Revenue, Moving Averages, YoY Growth, and Variances.
3. Use Power BI's native **Analytics Pane > Forecast** tool (90 points, 95% confidence, seasonality = 7).
4. Embed the custom Python visual using [`power_bi_python_visual.py`](power_bi_python_visual.py).

---

## 📄 Full Project Documentation
For the detailed internship submission report with business insights, see [`Task3_Predictive_Analytics_Report.md`](Task3_Predictive_Analytics_Report.md).
