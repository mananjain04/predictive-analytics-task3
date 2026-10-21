# Task 3: Predictive Analytics Using Historical Data
## Project Report & Technical Documentation

**Author / Candidate:** Data Analyst Intern  
**Program:** Power BI & Data Analytics Internship (Thiranex)  
**Task Title:** Predictive Analytics Using Historical Data  
**Task Objective:** Build a predictive model to forecast future trends.  
**Key Deliverables:** Cleaned dataset, Regression & Time-Series models, Model Accuracy Evaluation, Future Trend Forecast Visualizations, and Power BI Integration Guide.

---

## 1. Executive Summary

This project delivers an end-to-end predictive analytics and forecasting system designed to predict future business revenue trajectories based on historical performance data. Leveraging 5 full years (2021–2025; 1,826 daily records) of multi-dimensional retail transaction data, the system cleans, preprocesses, and models trend and seasonal dynamics using both parametric/non-parametric regression and univariate time-series algorithms.

### Key Highlights:
- **Data Preprocessing:** Handled duplicate records, imputed missing entries via forward/backward fill, removed negative sensor values, and clipped statistical revenue anomalies.
- **Multi-Model Strategy:** Developed and evaluated 5 distinct predictive models:
  1. Multi-variable Linear Regression
  2. Ridge Regularized Regression (L2)
  3. Random Forest Regressor
  4. Holt-Winters Triple Exponential Smoothing (Additive Trend + Weekly Seasonality)
  5. Seasonal Autoregressive Integrated Moving Average (SARIMA)
- **Model Evaluation:** Multi-variable Linear Regression achieved the lowest error on the holdout test set with an **MAE of $1,353.44**, **RMSE of $1,876.23**, a **MAPE of 2.60%**, and **R² = 0.8177** (explaining 81.8% of variance). Random Forest Regressor demonstrated strong non-linear feature capture with **R² = 0.6007** and **MAPE of 3.82%**.
- **Forward Horizon:** Produced a forward 90-day predictive forecast for Q1 2026 equipped with **95% Confidence Interval bounds** for risk management and scenario planning.
- **Power BI Readiness:** Formatted data into ready-to-import CSV and Excel structures, supplied copy-paste DAX formulas, detailed the Power BI Analytics Pane Forecasting tool configuration, and delivered an embedded Python Visual script.

---

## 2. Dataset Architecture & Preprocessing

### 2.1 Raw Dataset Profile
The initial raw dataset (`raw_historical_sales_data.csv`) simulates a high-growth multi-channel business with 1,831 records spanning 2021-01-01 through 2025-12-31.

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `Date` | Date (`YYYY-MM-DD`) | Daily transaction timeline |
| `Region` | Categorical | Geographic market (North, South, East, West) |
| `Category` | Categorical | Product category (Electronics, Home, Fashion, Beauty) |
| `Units_Sold` | Integer | Daily volume of items sold |
| `Avg_Unit_Price` | Decimal | Average selling price per item ($) |
| `Discount_Pct` | Decimal | Average promotional discount applied (0.02 - 0.35) |
| `Marketing_Spend` | Decimal | Daily omni-channel advertising budget ($) |
| `Foot_Traffic` | Integer | Daily store visitor and online portal sessions |
| `Holiday_Flag` | Binary (0/1) | Major holiday/promotional event indicator |
| `Economic_Index` | Decimal | Macro consumer confidence indicator |
| `Sales_Revenue` | Decimal | Target dependent variable ($) |

### 2.2 Data Cleaning Pipeline
Real-world operational datasets contain anomalies and omissions. The automated preprocessing pipeline executed the following steps:
1. **Deduplication:** Identified and eliminated 5 redundant duplicate rows.
2. **Invalid Value Correction:** Detected and rectified negative marketing expenditure records (`Marketing_Spend < 0`).
3. **Missing Value Imputation:** Detected 30 missing values in `Marketing_Spend` and 32 in `Foot_Traffic`. Imputed via forward fill (`ffill`) followed by backward fill (`bfill`) to preserve chronological continuity.
4. **Outlier Mitigation:** Applied Interquartile Range (IQR) capping on `Sales_Revenue` ($Q_1 - 2.5 \times IQR$ to $Q_3 + 2.5 \times IQR$) to eliminate extreme recording glitches while preserving genuine holiday surges.
5. **Feature Engineering:**
   - **Calendar Dynamics:** Year, Quarter, Month, Week of Year, Day of Week, Weekend Flag.
   - **Cyclical Encoding:** Sine and Cosine transformations of Day of Year ($\sin(2\pi d / 365.25)$, $\cos(2\pi d / 365.25)$) to capture smooth annual seasonality.
   - **Autoregressive Lags:** 1-day lag (`Sales_Lag_1`) and 7-day lag (`Sales_Lag_7`).
   - **Moving Statistics:** 7-day rolling average, 7-day rolling standard deviation, and 30-day rolling trend baseline.

---

## 3. Exploratory Data Analysis & Trend Diagnostics

![EDA Trend and Seasonality](visualizations/01_eda_trend_seasonality.png)

1. **Long-Term Trend:** The 30-day moving average reveals consistent compound annual growth, expanding from ~$18,000/day in early 2021 to over ~$50,000/day by late 2025.
2. **Weekly Seasonality:** Clear intra-week oscillations persist throughout the timeline, with Friday and Saturday generating 40–50% higher revenue than Monday and Tuesday.
3. **Annual Seasonality:** Revenue peaks sharply in Q4 (November and December) driven by Black Friday, Cyber Week, and holiday shopping, followed by a post-holiday dip in January/February.
4. **Correlation Insights:**
   - `Foot_Traffic` exhibits the highest positive correlation with `Sales_Revenue` ($r \approx 0.78$).
   - `Marketing_Spend` shows strong positive correlation ($r \approx 0.65$), indicating high advertising elasticity.
   - `Holiday_Flag` acts as an event amplifier, increasing baseline conversions significantly.

---

## 4. Predictive Modeling Framework

To address different business forecasting requirements, two distinct classes of predictive models were implemented:

### Class A: Driver-Based Regression Models
Ideal for scenario analysis and marketing return simulations where external inputs are controllable.
- **Linear Regression (Ordinary Least Squares):** Models the direct linear relationship between sales and marketing spend, foot traffic, discounts, holiday status, and autoregressive lags.
- **Ridge Regression (L2 Regularization):** Adds penalty $\alpha \sum \beta_j^2$ to minimize collinearity between rolling averages and lagged features.
- **Random Forest Regressor:** An ensemble of 120 decision trees capturing non-linear interactions, diminishing returns on advertising spend, and holiday amplification.

### Class B: Univariate Time-Series Models
Ideal for autonomous baseline projections when future driver variables (like future foot traffic) are unmeasured.
- **Holt-Winters Triple Exponential Smoothing:** Employs level ($\alpha$), damped additive trend ($\beta$, $\phi$), and weekly seasonal smoothing ($\gamma$) with a period $s = 7$.
- **SARIMA $(1, 1, 1) \times (1, 0, 1)_7$:** Captures stationary differenced momentum and weekly autoregression.

---

## 5. Model Evaluation & Accuracy Benchmark

Models were trained on historical records through October 2025 ($N=1,736$) and evaluated against an out-of-sample holdout test window of **90 days** (October–December 2025; $N=90$), matching a standard quarterly corporate forecasting cycle.

### 5.1 Comparative Performance Metrics

| Model | MAE ($) | RMSE ($) | MAPE (%) | $R^2$ Score | Key Strengths & Use Cases |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | **$1,353.44** | **$1,876.23** | **2.60%** | **0.8177** | **Best overall accuracy; high interpretability** |
| **Ridge Regression** | $1,356.09 | $1,890.27 | 2.61% | 0.8149 | Robust against feature multicollinearity |
| **Random Forest** | $2,067.47 | $2,776.55 | 3.82% | 0.6007 | Captures non-linear thresholds & variable importance |
| **Holt-Winters ES** | $4,049.22 | $4,497.75 | 7.79% | -0.0477 | Autonomous univariate forecast; requires no drivers |
| **SARIMA** | $4,737.05 | $5,278.88 | 9.24% | -0.4433 | Time-series momentum without exogenous assumptions |

![Model Accuracy Comparison](visualizations/04_model_accuracy_comparison.png)

### 5.2 Performance Analysis
- **Top Performer:** **Linear Regression** achieved exceptional predictive precision with an average error of only **2.60%** across the 90-day test horizon. The combination of rolling averages, autoregressive lags, and marketing drivers captured 81.8% of daily revenue variance.
- **Driver Analysis:** Random Forest feature importance analysis confirmed that **Rolling 30-Day Mean**, **Foot Traffic**, **Marketing Spend**, and **Holiday Flag** are the top 4 revenue drivers.

![Regression Predictions and Feature Importance](visualizations/02_regression_predictions.png)

---

## 6. Forward Trend Forecasting (Q1 2026)

Using the full 5-year historical training dataset, a 90-day forward projection was generated covering **January 1, 2026 to March 31, 2026**.

![Forward Time Series Forecast](visualizations/03_time_series_forecasting.png)

### Key Projections for Q1 2026:
- **Projected Revenue Range:** Daily sales are forecast to stabilize between **$48,500 and $59,200**, transitioning from post-holiday inventory normalization in January to steady spring acceleration in March.
- **Total Q1 2026 Expected Revenue:** Projected cumulative revenue of **~$4.85 Million**.
- **Confidence Intervals (95% CI):**
  - Expected Lower Bound: $44,100 / day
  - Expected Upper Bound: $63,800 / day
  - Band width accounts for weekend variance and baseline stochastic uncertainty.

---

## 7. Power BI Integration & Visuals Architecture

The project outputs are structured for native ingestion into Power BI Desktop:

1. **Unified Master Time Series Table (`power_bi_master_sales_forecast.csv`):** Connects actual historical revenue and predicted forward revenue continuously.
2. **Native Power BI Analytics Forecast:** Uses Power BI's built-in ETS engine configured with 90-day horizon, 95% confidence interval, and weekly seasonality ($s=7$).
3. **Embedded Python Visual:** A dedicated script (`power_bi_python_visual.py`) enables rendering the customized forecast visual directly onto the report canvas.
4. **Interactive Dashboard Blueprints:** Described in full in `Power_BI_Dashboard_Guide.md`, featuring Executive KPI cards, Driver Sensitivity scatter plots, and Scenario Planning sliders.

---

## 8. Strategic Business Recommendations

1. **Budget Optimization on Peak Days:** Shift 20–25% of weekday marketing budget into Thursday afternoon and Friday morning to maximize weekend conversion rates.
2. **Proactive Inventory Planning:** Utilize the 90-day forecast confidence upper bound ($63,800) for safety stock targets in high-turnover categories (Electronics and Apparel).
3. **Post-Holiday Retention:** Launch targeted loyalty campaigns during the late January seasonal dip to stabilize baseline revenue.
4. **Foot Traffic Conversion:** Because foot traffic is the leading predictive correlate ($r=0.78$), in-store conversion incentives (bundle discounts, digital vouchers) will produce the highest ROI.

---

## 9. Deliverables & Submission Verification

- [x] **Raw Historical Dataset:** `data/raw_historical_sales_data.csv`
- [x] **Cleaned Preprocessed Dataset:** `data/cleaned_historical_sales_data.csv`
- [x] **Master Power BI Dataset:** `data/power_bi_master_sales_forecast.csv`
- [x] **Excel Analytics Workbook:** `data/Sales_Predictive_Analytics_Data.xlsx`
- [x] **Model Evaluation Metrics Table:** `data/model_evaluation_metrics.csv`
- [x] **Future 90-Day Predictions Table:** `data/future_forecast_predictions.csv`
- [x] **Dataset Generator Script:** `generate_dataset.py`
- [x] **Machine Learning Pipeline Script:** `predictive_modeling.py`
- [x] **Power BI Python Visual Script:** `power_bi_python_visual.py`
- [x] **Power BI Implementation Guide:** `Power_BI_Dashboard_Guide.md`
- [x] **Project Visualizations (4 High-Res Plots):** `visualizations/*.png`
- [x] **Comprehensive Project Report:** `Task3_Predictive_Analytics_Report.md`
