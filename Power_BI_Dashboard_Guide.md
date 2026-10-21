# Power BI Implementation & Dashboard Guide: Predictive Analytics

This guide provides step-by-step instructions to load, model, and visualize the predictive analytics dataset in **Power BI Desktop**.

---

## 1. Data Import

1. Open **Power BI Desktop**.
2. Click **Get Data** > **Text/CSV** (or **Excel Workbook**).
3. Select either:
   - `data/cleaned_historical_sales_data.csv` (Primary historical data with engineered features)
   - `data/power_bi_master_sales_forecast.csv` (Unified historical actuals + 90-day forecast)
   - `data/Sales_Predictive_Analytics_Data.xlsx` (Multi-sheet workbook containing all tables)
4. Click **Load** (or **Transform Data** in Power Query to inspect column types).
5. Verify column data types:
   - `Date`: Date
   - `Sales_Revenue`, `Actual_Sales_Revenue`, `Forecasted_Sales_Revenue`: Decimal Number (Currency `$`)
   - `Marketing_Spend`, `Foot_Traffic`, `Units_Sold`: Whole/Decimal Number
   - `Discount_Pct`: Percentage (`%`)
   - `Region`, `Category`, `Series_Type`: Text

---

## 2. Essential DAX Measures

Create a dedicated `_Measures` table in Power BI and add the following DAX calculations:

### 1. Total Sales Revenue
```dax
Total Sales Revenue = 
SUM(cleaned_historical_sales_data[Sales_Revenue])
```

### 2. 30-Day Moving Average Revenue
```dax
Revenue 30D Moving Avg = 
AVERAGEX(
    DATESINPERIOD(
        cleaned_historical_sales_data[Date],
        LASTDATE(cleaned_historical_sales_data[Date]),
        -30,
        DAY
    ),
    [Total Sales Revenue]
)
```

### 3. Total Marketing Spend
```dax
Total Marketing Spend = 
SUM(cleaned_historical_sales_data[Marketing_Spend])
```

### 4. Return on Marketing Spend (ROAS / Efficiency Ratio)
```dax
Marketing ROI = 
DIVIDE([Total Sales Revenue], [Total Marketing Spend], 0)
```

### 5. Year-over-Year (YoY) Sales Growth %
```dax
YoY Sales Growth % = 
VAR CurrentYearSales = [Total Sales Revenue]
VAR PriorYearSales = CALCULATE([Total Sales Revenue], SAMEPERIODLASTYEAR(cleaned_historical_sales_data[Date]))
RETURN
DIVIDE(CurrentYearSales - PriorYearSales, PriorYearSales, 0)
```

### 6. Forecast vs Actual Variance
```dax
Forecast Variance = 
SUM(power_bi_master_sales_forecast[Actual_Sales_Revenue]) - SUM(power_bi_master_sales_forecast[Forecasted_Sales_Revenue])
```

---

## 3. Power BI Native Forecasting Tool (Analytics Pane)

Power BI Desktop includes a built-in Exponential Smoothing (ETS) forecast engine that runs directly inside Line Charts without writing code:

### Step-by-Step Configuration:
1. Insert a standard **Line Chart** visual onto your report canvas.
2. In the Visualizations fields:
   - Drag `Date` to the **X-axis** (ensure it displays as continuous `Date` rather than Date Hierarchy).
   - Drag `Sales_Revenue` to the **Y-axis** (Aggregation: `Sum`).
3. Select the visual and click the **Analytics Pane** (magnifying glass icon next to Format Visual).
4. Scroll down to **Forecast** and click **Add**:
   - **Forecast Length**: `90` Points (Days)
   - **Ignore Last**: `0` Points (or set to `90` to evaluate backtesting against actuals)
   - **Confidence Interval**: `95%`
   - **Seasonality**: `7` Points (matches the weekly retail shopping cycle)
5. Power BI automatically projects the forward trajectory line with upper and lower shaded confidence cones!

---

## 4. Power BI Python Visual Integration

You can also render the exact custom Python machine learning forecast directly inside Power BI:

1. Enable Python scripting in Power BI:
   - Go to **File** > **Options and settings** > **Options** > **Python scripting**.
   - Verify that your detected Python environment (Python 3.14) is selected.
2. In the Visualizations pane, click the **Python Visual** icon.
3. Select the visual, and drag `Date` and `Sales_Revenue` into the **Values** field well.
4. Copy and paste the entire script from `power_bi_python_visual.py` into the Python Script Editor at the bottom of Power BI Desktop.
5. Click the **Run script** button (play icon) at the top of the script editor.
6. A high-resolution matplotlib chart will render natively inside your Power BI canvas!

---

## 5. Recommended 3-Page Dashboard Layout

### Page 1: Executive KPI & Trend Overview
- **Top Row**: 4 KPI Cards:
  - Total Historical Revenue ($)
  - Average Daily Revenue ($)
  - YoY Growth %
  - Marketing ROI (ROAS)
- **Main Section**: Line Chart of Daily Revenue with 30-Day Rolling Trend Line and Trend Line toggle.
- **Side Panel**: Slicers for `Region`, `Product Category`, and `Date Range`.
- **Bottom Section**: Column chart showing Revenue by Product Category & Donut chart of Revenue by Region.

### Page 2: Predictive Driver Analysis (Regression)
- **Top Left**: Scatter Plot: `Marketing_Spend` (X-axis) vs `Sales_Revenue` (Y-axis) with Trend Line.
- **Top Right**: Scatter Plot: `Foot_Traffic` (X-axis) vs `Sales_Revenue` (Y-axis) with Trend Line.
- **Bottom Left**: Clustered Bar Chart of Feature Importance Scores (imported from `02_regression_predictions.png` or table).
- **Bottom Right**: Key Influencers Visual (Power BI AI Visual) analyzing factors that cause `Sales_Revenue` to increase.

### Page 3: Forward 90-Day Forecasting & Scenario Planning
- **Main Visual**: Unified Line Chart from `power_bi_master_sales_forecast.csv`:
  - Blue line for `Actual_Sales_Revenue`
  - Red dashed line for `Forecasted_Sales_Revenue`
  - Shaded band for `Confidence_Lower_95` and `Confidence_Upper_95`
- **Right Panel**: Summary Matrix of Q1 2026 Expected Revenue ($), Lower Bound ($), and Upper Bound ($).
- **Embedded Python Visual**: Native interactive time-series forecast plot.
