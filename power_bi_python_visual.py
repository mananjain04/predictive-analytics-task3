# ==============================================================================
# POWER BI PYTHON VISUAL SCRIPT: PREDICTIVE TIME-SERIES FORECASTING
# ==============================================================================
# Instructions for Power BI Desktop:
# 1. In Power BI Desktop, click on the "Python visual" icon in the Visualizations pane.
# 2. Drag the following fields into the 'Values' well:
#    - Date
#    - Sales_Revenue (or Actual_Sales_Revenue)
# 3. Paste this code into the Power BI Python Script Editor and click 'Run'.
# ==============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Power BI provides the selected data in a pandas DataFrame called 'dataset'
# Deduplicate and sort by Date
df = dataset.drop_duplicates().copy()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Fit Exponential Smoothing with Additive Trend and Weekly Seasonality
sales_series = df['Sales_Revenue'].dropna()
hw_model = ExponentialSmoothing(
    sales_series,
    trend='add',
    seasonal='add',
    seasonal_periods=7,
    damped_trend=True
).fit()

# Forecast forward 60 days
forecast_steps = 60
last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_vals = hw_model.forecast(forecast_steps).values

# Empirical 95% Confidence Band
residuals = sales_series.iloc[-len(hw_model.fittedvalues):] - hw_model.fittedvalues
sigma = np.std(residuals)
margin = 1.96 * sigma * np.sqrt(1 + (np.arange(forecast_steps) / forecast_steps) * 0.4)

lower_bound = np.maximum(0, forecast_vals - margin)
upper_bound = forecast_vals + margin

# Render Visual
plt.figure(figsize=(10, 5), dpi=150)
recent_df = df.iloc[-120:]
plt.plot(recent_df['Date'], recent_df['Sales_Revenue'], label='Actual Sales Revenue', color='#1E40AF', lw=1.8)
plt.plot(future_dates, forecast_vals, label='60-Day Forward Forecast', color='#DC2626', lw=2.2)
plt.fill_between(future_dates, lower_bound, upper_bound, color='#F87171', alpha=0.3, label='95% Confidence Interval')

plt.axvline(x=last_date, color='grey', linestyle='--', lw=1)
plt.title('Predictive Trend Forecast with Confidence Interval', fontsize=12, fontweight='bold')
plt.xlabel('Date', fontsize=10)
plt.ylabel('Revenue ($)', fontsize=10)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
