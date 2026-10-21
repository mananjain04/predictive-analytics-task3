import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set high-quality styling for matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def clean_and_preprocess_data(raw_path="data/raw_historical_sales_data.csv", output_dir="data"):
    print("=" * 70)
    print("STEP 1: DATA CLEANING AND PREPROCESSING")
    print("=" * 70)
    
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    print(f"Loaded raw dataset with {initial_rows} records.")

    # 1. Deduplication
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"[x] Removed {duplicates_removed} duplicate records.")

    # 2. Date parsing and ordering
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 3. Clean invalid values (e.g. negative marketing spend)
    invalid_mktg = (df['Marketing_Spend'] < 0)
    if invalid_mktg.any():
        print(f"[x] Fixed {invalid_mktg.sum()} negative marketing spend entry.")
        df.loc[invalid_mktg, 'Marketing_Spend'] = np.nan

    # 4. Handle Missing Values
    null_counts = df.isnull().sum()
    print(f"Missing values detected before imputation:\n{null_counts[null_counts > 0]}")

    # Forward fill then backward fill for time-series continuity
    df['Marketing_Spend'] = df['Marketing_Spend'].ffill().bfill()
    df['Foot_Traffic'] = df['Foot_Traffic'].ffill().bfill()

    # 5. Outlier Detection and Capping on Target (Sales_Revenue)
    Q1 = df['Sales_Revenue'].quantile(0.25)
    Q3 = df['Sales_Revenue'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    outliers = (df['Sales_Revenue'] < lower_bound) | (df['Sales_Revenue'] > upper_bound)
    n_outliers = outliers.sum()
    print(f"[x] Identified {n_outliers} statistical outliers in Sales_Revenue. Capped to normal bounds.")
    df.loc[df['Sales_Revenue'] > upper_bound, 'Sales_Revenue'] = upper_bound
    df.loc[df['Sales_Revenue'] < lower_bound, 'Sales_Revenue'] = lower_bound

    # 6. Feature Engineering (Temporal & Lag features)
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day_Name'] = df['Date'].dt.strftime('%a')
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Cyclical seasonal encoding for annual patterns
    df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # Lags and Rolling Windows for daily sales
    df['Sales_Lag_1'] = df['Sales_Revenue'].shift(1)
    df['Sales_Lag_7'] = df['Sales_Revenue'].shift(7)
    df['Rolling_Mean_7'] = df['Sales_Revenue'].shift(1).rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Sales_Revenue'].shift(1).rolling(window=7).std()
    df['Rolling_Mean_30'] = df['Sales_Revenue'].shift(1).rolling(window=30).mean()

    # Backfill the initial rolling/lag rows
    df = df.bfill()

    clean_path = os.path.join(output_dir, "cleaned_historical_sales_data.csv")
    df.to_csv(clean_path, index=False)
    print(f"[+] Successfully exported cleaned dataset to: {clean_path} ({len(df)} records)\n")
    return df

def generate_eda_visualizations(df, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS & TREND DECOMPOSITION")
    print("=" * 70)

    # Set up 2x2 multi-panel EDA plot
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. 5-Year Historical Sales with 30-Day Moving Average
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Date'], df['Sales_Revenue'], color='#93C5FD', alpha=0.6, lw=0.9, label='Daily Actual Sales')
    ax1.plot(df['Date'], df['Rolling_Mean_30'], color='#1E40AF', lw=2.2, label='30-Day Moving Average (Trend)')
    ax1.set_title('Historical Sales Revenue Trajectory & 30-Day Rolling Trend (2021-2025)', fontsize=13, fontweight='bold', pad=8)
    ax1.set_ylabel('Sales Revenue ($)', fontsize=11)
    ax1.legend(loc='upper left', frameon=True)

    # 2. Monthly Seasonality Boxplot
    ax2 = fig.add_subplot(gs[1, 0])
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.boxplot(data=df, x='Month_Name', y='Sales_Revenue', order=month_order, palette='crest', ax=ax2, fliersize=2)
    ax2.set_title('Seasonal Revenue Distribution by Month', fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Sales Revenue ($)', fontsize=11)

    # 3. Correlation Heatmap of Key Predictive Drivers
    ax3 = fig.add_subplot(gs[1, 1])
    corr_cols = ['Sales_Revenue', 'Marketing_Spend', 'Foot_Traffic', 'Discount_Pct', 'Holiday_Flag', 'Economic_Index', 'Rolling_Mean_7']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=ax3, annot_kws={"size": 9})
    ax3.set_title('Feature Correlation Matrix with Sales Revenue', fontsize=12, fontweight='bold', pad=8)
    ax3.tick_params(axis='x', rotation=30)

    eda_path = os.path.join(output_dir, "01_eda_trend_seasonality.png")
    plt.savefig(eda_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved EDA Analysis Visual: {eda_path}")

def train_and_evaluate_models(df, output_dir="visualizations"):
    print("=" * 70)
    print("STEP 3: PREDICTIVE MODELING & ACCURACY EVALUATION")
    print("=" * 70)

    # Holdout evaluation: 90-day test set (matching standard quarterly business planning)
    test_horizon = 90
    train_df = df.iloc[:-test_horizon].copy()
    test_df = df.iloc[-test_horizon:].copy()

    print(f"Training set: {len(train_df)} records ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Testing set:  {len(test_df)} records ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    # Feature selection for Regression models
    feature_cols = [
        'Marketing_Spend', 'Foot_Traffic', 'Discount_Pct', 'Holiday_Flag',
        'Economic_Index', 'Is_Weekend', 'DayOfWeek', 'Sin_DayOfYear', 'Cos_DayOfYear',
        'Sales_Lag_1', 'Sales_Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30'
    ]
    target_col = 'Sales_Revenue'

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model_results = {}
    predictions_dict = {'Date': test_df['Date'], 'Actual': y_test.values}

    # 1. Multi-variable Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    model_results['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    predictions_dict['Linear Regression'] = y_pred_lr

    # 2. Ridge Regression
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    model_results['Ridge Regression'] = {
        'MAE': mean_absolute_error(y_test, y_pred_ridge),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred_ridge),
        'R2': r2_score(y_test, y_pred_ridge)
    }
    predictions_dict['Ridge Regression'] = y_pred_ridge

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    model_results['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }
    predictions_dict['Random Forest'] = y_pred_rf

    # 4. Holt-Winters Triple Exponential Smoothing (Additive Trend + Weekly Seasonality)
    hw_model = ExponentialSmoothing(
        train_df['Sales_Revenue'],
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        damped_trend=True,
        initialization_method='estimated'
    ).fit()
    y_pred_hw = hw_model.forecast(test_horizon).values
    model_results['Holt-Winters ES'] = {
        'MAE': mean_absolute_error(y_test, y_pred_hw),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_hw)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred_hw),
        'R2': r2_score(y_test, y_pred_hw)
    }
    predictions_dict['Holt-Winters ES'] = y_pred_hw

    # 5. SARIMA Time Series Model
    print("Fitting SARIMA model...")
    sarima_model = SARIMAX(
        train_df['Sales_Revenue'],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    y_pred_sarima = sarima_model.forecast(steps=test_horizon).values
    model_results['SARIMA'] = {
        'MAE': mean_absolute_error(y_test, y_pred_sarima),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_sarima)),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred_sarima),
        'R2': r2_score(y_test, y_pred_sarima)
    }
    predictions_dict['SARIMA'] = y_pred_sarima

    # Summary Table
    results_df = pd.DataFrame(model_results).T
    results_df = results_df.round({'MAE': 2, 'RMSE': 2, 'MAPE (%)': 2, 'R2': 4})
    print("\n" + "=" * 70)
    print("MODEL ACCURACY & EVALUATION SUMMARY METRICS")
    print("=" * 70)
    print(results_df)

    results_csv_path = os.path.join("data", "model_evaluation_metrics.csv")
    results_df.to_csv(results_csv_path)
    print(f"\n[+] Saved evaluation metrics to: {results_csv_path}")

    # Visualizations: 02_regression_predictions.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Actual vs Predicted (Random Forest / Linear Regression)
    axes[0].scatter(y_test, y_pred_lr, alpha=0.7, color='#2563EB', edgecolors='k', s=35, label='Linear Regression')
    axes[0].scatter(y_test, y_pred_rf, alpha=0.7, color='#10B981', edgecolors='k', s=35, label='Random Forest')
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (Y=X)')
    axes[0].set_title(f'Regression Models: Actual vs Predicted (LR R² = {model_results["Linear Regression"]["R2"]:.3f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Sales Revenue ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Sales Revenue ($)', fontsize=11)
    axes[0].legend(loc='upper left')

    # Feature Importance (Random Forest)
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    feat_imp.plot(kind='barh', ax=axes[1], color='#059669', edgecolor='black', alpha=0.85)
    axes[1].set_title('Feature Importance for Predictive Forecasts (Random Forest)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Relative Importance Score', fontsize=11)

    plt.tight_layout()
    reg_vis_path = os.path.join(output_dir, "02_regression_predictions.png")
    plt.savefig(reg_vis_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Regression Analysis Visual: {reg_vis_path}")

    # Visualizations: 04_model_accuracy_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    models = list(results_df.index)
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']

    # MAE
    axes[0, 0].bar(models, results_df['MAE'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title('Mean Absolute Error (MAE) - Lower is Better', fontweight='bold')
    axes[0, 0].set_ylabel('Error ($)')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # RMSE
    axes[0, 1].bar(models, results_df['RMSE'], color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_title('Root Mean Squared Error (RMSE) - Lower is Better', fontweight='bold')
    axes[0, 1].set_ylabel('Error ($)')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # MAPE
    axes[1, 0].bar(models, results_df['MAPE (%)'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE %) - Lower is Better', fontweight='bold')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # R2
    axes[1, 1].bar(models, results_df['R2'], color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_title('R² Score (Goodness of Fit) - Higher is Better', fontweight='bold')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    acc_vis_path = os.path.join(output_dir, "04_model_accuracy_comparison.png")
    plt.savefig(acc_vis_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Model Accuracy Comparison Plot: {acc_vis_path}")

    return rf, hw_model, sarima_model, results_df, test_df, predictions_dict

def generate_future_forecast(df, output_dir="data", vis_dir="visualizations"):
    print("=" * 70)
    print("STEP 4: GENERATING 90-DAY OUT-OF-SAMPLE FUTURE FORECAST (Q1 2026)")
    print("=" * 70)

    # Fit Holt-Winters on the entire 5-year cleaned historical dataset to project Q1 2026
    full_hw_model = ExponentialSmoothing(
        df['Sales_Revenue'],
        trend='add',
        seasonal='add',
        seasonal_periods=7,
        damped_trend=True,
        initialization_method='estimated'
    ).fit()

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq='D')
    
    # 90-day forecast
    forecast_values = np.round(full_hw_model.forecast(90).values, 2)
    
    # Empirical standard deviation of in-sample residuals
    residuals = df['Sales_Revenue'] - full_hw_model.fittedvalues
    residual_std = np.std(residuals)

    # 95% Confidence Interval band: 1.96 * residual_std * sqrt(1 + h/H)
    horizon_scale = np.sqrt(1 + (np.arange(90) / 90) * 0.4)
    margin = 1.96 * residual_std * horizon_scale
    lower_bound = np.round(np.maximum(0, forecast_values - margin), 2)
    upper_bound = np.round(forecast_values + margin, 2)

    future_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Forecasted_Sales_Revenue': forecast_values,
        'Confidence_Lower_95': lower_bound,
        'Confidence_Upper_95': upper_bound,
        'Forecast_Period': 'Q1 2026 Forward Outlook',
        'Trend_Signal': np.where(forecast_values > forecast_values[0], 'Expansion', 'Contraction')
    })

    future_path = os.path.join(output_dir, "future_forecast_predictions.csv")
    future_df.to_csv(future_path, index=False)
    print(f"[+] Saved future predictions to: {future_path}")

    # Build master Power BI time series table (Unified Actuals + Forecast)
    hist_subset = pd.DataFrame({
        'Date': df['Date'].dt.strftime('%Y-%m-%d'),
        'Actual_Sales_Revenue': df['Sales_Revenue'],
        'Forecasted_Sales_Revenue': np.nan,
        'Confidence_Lower_95': np.nan,
        'Confidence_Upper_95': np.nan,
        'Series_Type': 'Historical Actual'
    })
    
    future_subset = pd.DataFrame({
        'Date': future_df['Date'],
        'Actual_Sales_Revenue': np.nan,
        'Forecasted_Sales_Revenue': future_df['Forecasted_Sales_Revenue'],
        'Confidence_Lower_95': future_df['Confidence_Lower_95'],
        'Confidence_Upper_95': future_df['Confidence_Upper_95'],
        'Series_Type': 'Forecast Projection'
    })

    # Link the bridge point so lines connect continuously in Power BI
    future_subset.loc[0, 'Actual_Sales_Revenue'] = hist_subset.iloc[-1]['Actual_Sales_Revenue']

    master_pbi = pd.concat([hist_subset, future_subset], ignore_index=True)
    master_pbi_path = os.path.join(output_dir, "power_bi_master_sales_forecast.csv")
    master_pbi.to_csv(master_pbi_path, index=False)
    print(f"[+] Saved Power BI Master Timeseries Table: {master_pbi_path}")

    # Plot 03: Time Series Forecast Visual
    plt.figure(figsize=(15, 6))
    
    # Show last 180 days of actuals for sharp context
    recent_hist = df.iloc[-180:]
    plt.plot(recent_hist['Date'], recent_hist['Sales_Revenue'], label='Historical Actual Sales Revenue', color='#1E40AF', lw=1.8)
    
    # Forecast curve
    plt.plot(future_dates, forecast_values, label='90-Day Predictive Trend Forecast (Q1 2026)', color='#DC2626', lw=2.2, linestyle='-')
    plt.fill_between(future_dates, lower_bound, upper_bound, color='#FCA5A5', alpha=0.35, label='95% Confidence Interval')
    
    plt.axvline(x=last_date, color='black', linestyle=':', lw=1.5, label='Forecast Boundary (2025-12-31)')

    plt.title('Predictive Analytics: Historical Sales Trajectory & 90-Day Forward Trend Forecast', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date Timeline', fontsize=12)
    plt.ylabel('Sales Revenue ($)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    forecast_vis_path = os.path.join(vis_dir, "03_time_series_forecasting.png")
    plt.savefig(forecast_vis_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Forecast Visual: {forecast_vis_path}")

if __name__ == "__main__":
    df_clean = clean_and_preprocess_data()
    generate_eda_visualizations(df_clean)
    rf_model, hw_model, sarima_model, metrics_df, test_df, preds = train_and_evaluate_models(df_clean)
    generate_future_forecast(df_clean)
    print("\n[SUCCESS] ALL PREDICTIVE ANALYTICS PIPELINE STAGES COMPLETED!")
