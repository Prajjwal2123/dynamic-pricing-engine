import pandas as pd
import numpy as np
import os

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel dataset from the given path."""
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def generate_synthetic_data(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic pricing dataset.
    Use this if you don't have a real dataset yet.
    """
    np.random.seed(seed)

    product_ids = [f"P{str(i).zfill(4)}" for i in range(1, 201)]
    categories  = ["Electronics", "Clothing", "Groceries", "Furniture", "Books"]

    product_id       = np.random.choice(product_ids, n_rows)
    category         = np.random.choice(categories, n_rows)
    base_price       = np.round(np.random.uniform(50, 5000, n_rows), 2)

    # Competitor price: within ±20% of base price
    competitor_price = np.round(base_price * np.random.uniform(0.80, 1.20, n_rows), 2)

    # Inventory: 0 to 500 units
    inventory        = np.random.randint(0, 500, n_rows)
    inventory_ratio  = np.round(inventory / 500, 4)

    # Temporal features
    days             = pd.date_range(start="2023-01-01", periods=n_rows, freq="H")
    day_of_week      = days.dayofweek          # 0=Monday
    is_weekend       = (day_of_week >= 5).astype(int)
    month            = days.month
    is_holiday       = np.random.choice([0, 1], n_rows, p=[0.93, 0.07])

    # Demand: influenced by price, inventory, and some noise
    # Lower price → higher demand. Low inventory → slightly higher demand (scarcity effect)
    price_effect     = 1 - (base_price / 5000) * 0.4
    inventory_effect = 1 + (1 - inventory_ratio) * 0.2
    noise            = np.random.normal(1.0, 0.1, n_rows)
    demand           = np.round(
        100 * price_effect * inventory_effect * noise * (1 + is_weekend * 0.15)
    ).clip(0).astype(int)

    # Historical sales: correlated with demand + random variance
    historical_sales_7d  = np.round(demand * np.random.uniform(0.85, 1.15, n_rows)).astype(int)
    historical_sales_30d = np.round(demand * 4 * np.random.uniform(0.90, 1.10, n_rows)).astype(int)

    df = pd.DataFrame({
        "timestamp":            days,
        "product_id":           product_id,
        "category":             category,
        "base_price":           base_price,
        "competitor_price":     competitor_price,
        "inventory":            inventory,
        "inventory_ratio":      inventory_ratio,
        "day_of_week":          day_of_week,
        "is_weekend":           is_weekend,
        "month":                month,
        "is_holiday":           is_holiday,
        "demand":               demand,
        "historical_sales_7d":  historical_sales_7d,
        "historical_sales_30d": historical_sales_30d,
    })

    print(f"Generated synthetic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_data_quality_report(df: pd.DataFrame) -> None:
    """Print a quick data quality summary."""
    print("\n===== DATA QUALITY REPORT =====")
    print(f"Shape          : {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates     : {df.duplicated().sum()}")
    print(f"\nNumeric summary:\n{df.describe().round(2)}")
    print("================================\n")


if __name__ == "__main__":
    # Generate and save synthetic data
    df = generate_synthetic_data(n_rows=10000)
    basic_data_quality_report(df)

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/synthetic_pricing_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")