import os
import numpy as np
import pandas as pd

def generate_historical_sales_data(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 5 full years of daily records: 2021-01-01 to 2025-12-31 (1826 days)
    date_range = pd.date_range(start="2021-01-01", end="2025-12-31", freq="D")
    n_days = len(date_range)

    # Base trend (upward linear growth over 5 years)
    t = np.arange(n_days)
    trend = 12000 + 15 * t + 0.002 * (t ** 2)

    # Multi-frequency seasonality
    # Annual seasonality (peak in Nov-Dec holiday season, dip in Jan-Feb)
    day_of_year = date_range.dayofyear.values
    annual_season = 3500 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25) + \
                    2000 * np.sin(4 * np.pi * (day_of_year - 30) / 365.25)
    
    # Weekly seasonality (higher sales on Fri, Sat, Sun)
    day_of_week = date_range.dayofweek.values  # 0=Mon, 6=Sun
    weekly_season = np.array([
        -1200 if d == 0 else  # Monday dip
        -800  if d == 1 else  # Tuesday
        -200  if d == 2 else  # Wednesday
         400  if d == 3 else  # Thursday
        1500  if d == 4 else  # Friday
        2800  if d == 5 else  # Saturday surge
        2200                  # Sunday
        for d in day_of_week
    ])

    # Holiday spikes (Black Friday, Cyber Monday, Christmas, New Year, Mid-year clearance)
    is_holiday = np.zeros(n_days, dtype=int)
    holiday_boost = np.zeros(n_days)
    for i, date in enumerate(date_range):
        m, d = date.month, date.day
        # New Year
        if (m == 1 and d in [1, 2]) or (m == 12 and d in [24, 25, 26, 31]):
            is_holiday[i] = 1
            holiday_boost[i] = 4500
        # Black Friday / Cyber Monday (late November)
        elif m == 11 and 23 <= d <= 30:
            is_holiday[i] = 1
            holiday_boost[i] = 6000
        # Summer / Mid-year sale (July 4th / Prime week)
        elif m == 7 and 10 <= d <= 15:
            is_holiday[i] = 1
            holiday_boost[i] = 3000
        # Random public holidays (approx 8 days/year)
        elif np.random.rand() < 0.02:
            is_holiday[i] = 1
            holiday_boost[i] = 2000

    # Marketing Spend ($800 - $5,000 daily with higher spend around holidays)
    base_mktg = 1200 + 0.5 * t
    mktg_spend = base_mktg + (holiday_boost * 0.35) + np.random.normal(0, 300, n_days)
    mktg_spend = np.clip(mktg_spend, 500, 8000)

    # Foot Traffic / Web Visitors
    foot_traffic = 2500 + 0.8 * t + 0.4 * weekly_season + 0.25 * holiday_boost + np.random.normal(0, 200, n_days)
    foot_traffic = np.clip(foot_traffic, 1000, 15000).astype(int)

    # Discount Percentage (5% to 25%)
    discount_pct = np.clip(0.08 + 0.08 * is_holiday + np.random.normal(0, 0.03, n_days), 0.02, 0.35)

    # Economic Sentiment Index (scale 80 - 120)
    economic_index = 100 + 10 * np.sin(2 * np.pi * t / (365.25 * 2)) + np.random.normal(0, 2, n_days)

    # Composite Sales Revenue
    noise = np.random.normal(0, 1000, n_days)
    revenue = trend + annual_season + weekly_season + holiday_boost + (mktg_spend * 1.8) + (foot_traffic * 1.2) + noise
    revenue = np.clip(revenue, 5000, None)

    # Units Sold & Average Unit Price
    avg_unit_price = np.round(np.random.normal(55, 6, n_days), 2)
    units_sold = np.round(revenue / (avg_unit_price * (1 - discount_pct * 0.5))).astype(int)

    # Recalculate true Sales Revenue
    sales_revenue = np.round(revenue, 2)

    # Categories & Regions
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Home & Lifestyle", "Fashion & Apparel", "Beauty & Care"]
    
    # Primary aggregated time series table
    df_raw = pd.DataFrame({
        "Date": date_range.strftime("%Y-%m-%d"),
        "Region": np.random.choice(regions, n_days, p=[0.3, 0.25, 0.25, 0.2]),
        "Category": np.random.choice(categories, n_days, p=[0.35, 0.25, 0.25, 0.15]),
        "Units_Sold": units_sold,
        "Avg_Unit_Price": avg_unit_price,
        "Discount_Pct": np.round(discount_pct, 3),
        "Marketing_Spend": np.round(mktg_spend, 2),
        "Foot_Traffic": foot_traffic,
        "Holiday_Flag": is_holiday,
        "Economic_Index": np.round(economic_index, 2),
        "Sales_Revenue": sales_revenue
    })

    # Introduce real-world data imperfections in the RAW dataset to demonstrate cleaning:
    # 1. Random missing values in Marketing_Spend and Foot_Traffic (~2% missing)
    mask_nan_mktg = np.random.rand(n_days) < 0.02
    df_raw.loc[mask_nan_mktg, "Marketing_Spend"] = np.nan

    mask_nan_traffic = np.random.rand(n_days) < 0.015
    df_raw.loc[mask_nan_traffic, "Foot_Traffic"] = np.nan

    # 2. Add 5 duplicate rows
    duplicate_rows = df_raw.sample(5, random_state=101)
    df_raw = pd.concat([df_raw, duplicate_rows], ignore_index=True)

    # 3. Add 2 outlier anomalies (e.g., sensor glitch / data entry error)
    df_raw.loc[120, "Sales_Revenue"] = df_raw.loc[120, "Sales_Revenue"] * 8.5  # Extreme outlier
    df_raw.loc[450, "Marketing_Spend"] = -500.0  # Invalid negative spend entry

    raw_path = os.path.join(output_dir, "raw_historical_sales_data.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"Generated raw dataset: {raw_path} with {len(df_raw)} records.")
    return df_raw

if __name__ == "__main__":
    generate_historical_sales_data()
