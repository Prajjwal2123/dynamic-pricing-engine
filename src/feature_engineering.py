import pandas as pd
import numpy as np
import os


def load_raw_data(filepath: str = "data/raw/synthetic_pricing_data.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 1. DEMAND FEATURES
# ─────────────────────────────────────────────
def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling average demand per product — captures recent demand trend."""
    df = df.sort_values(["product_id", "timestamp"])

    df["demand_rolling_7d"] = (
        df.groupby("product_id")["demand"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
        .round(2)
    )

    df["demand_rolling_30d"] = (
        df.groupby("product_id")["demand"]
        .transform(lambda x: x.rolling(window=30, min_periods=1).mean().shift(1))
        .round(2)
    )

    # Demand momentum: is recent demand rising or falling?
    df["demand_momentum"] = (df["demand_rolling_7d"] - df["demand_rolling_30d"]).round(2)

    print("✓ Demand features added")
    return df


# ─────────────────────────────────────────────
# 2. INVENTORY FEATURES
# ─────────────────────────────────────────────
def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inventory ratio and scarcity flags."""

    # Already have inventory_ratio — add urgency tiers
    df["stock_tier"] = pd.cut(
        df["inventory_ratio"],
        bins=[0, 0.10, 0.25, 0.50, 1.0],
        labels=["critical", "low", "medium", "high"],
        include_lowest=True
    )

    df["is_low_stock"]      = (df["inventory_ratio"] < 0.25).astype(int)
    df["is_critical_stock"] = (df["inventory_ratio"] < 0.10).astype(int)

    print("✓ Inventory features added")
    return df


# ─────────────────────────────────────────────
# 3. COMPETITOR PRICING FEATURES
# ─────────────────────────────────────────────
def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """How we're positioned vs the competitor."""

    df["price_gap"]         = (df["base_price"] - df["competitor_price"]).round(2)
    df["price_gap_pct"]     = ((df["price_gap"] / df["competitor_price"]) * 100).round(2)
    df["is_cheaper"]        = (df["base_price"] < df["competitor_price"]).astype(int)
    df["is_overpriced"]     = (df["price_gap_pct"] > 20).astype(int)  # >20% more expensive

    # Rolling competitor price trend (are they raising or dropping prices?)
    df = df.sort_values(["product_id", "timestamp"])
    df["competitor_price_trend"] = (
        df.groupby("product_id")["competitor_price"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
        .round(2)
    )
    df["competitor_trending_up"] = (
        df["competitor_price"] > df["competitor_price_trend"]
    ).astype(int)

    print("✓ Competitor features added")
    return df


# ─────────────────────────────────────────────
# 4. TEMPORAL FEATURES
# ─────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based signals — seasonality, peaks."""

    df["hour"]         = df["timestamp"].dt.hour
    df["quarter"]      = df["timestamp"].dt.quarter

    # Peak hours: 10am–1pm and 6pm–9pm
    df["is_peak_hour"] = df["hour"].isin(range(10, 14)).astype(int) | \
                         df["hour"].isin(range(18, 22)).astype(int)

    # End of month buying spike
    df["is_month_end"] = (df["timestamp"].dt.day >= 28).astype(int)

    # Season
    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3:  "spring", 4: "spring", 5: "spring",
        6:  "summer", 7: "summer", 8: "summer",
        9:  "autumn", 10: "autumn", 11: "autumn"
    })

    print("✓ Temporal features added")
    return df


# ─────────────────────────────────────────────
# 5. PRICE ELASTICITY PROXY
# ─────────────────────────────────────────────
def add_elasticity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate how sensitive demand is to price changes.
    Elasticity = % change in demand / % change in price (lagged).
    """
    df = df.sort_values(["product_id", "timestamp"])

    df["price_lag1"]  = df.groupby("product_id")["base_price"].shift(1)
    df["demand_lag1"] = df.groupby("product_id")["demand"].shift(1)

    pct_price_change  = (df["base_price"]  - df["price_lag1"])  / (df["price_lag1"]  + 1e-9)
    pct_demand_change = (df["demand"]      - df["demand_lag1"]) / (df["demand_lag1"] + 1e-9)

    df["price_elasticity"] = (pct_demand_change / (pct_price_change + 1e-9)).round(4)

    # Cap extreme values (outliers from near-zero price changes)
    df["price_elasticity"] = df["price_elasticity"].clip(-10, 10)

    # Drop helper columns
    df.drop(columns=["price_lag1", "demand_lag1"], inplace=True)

    print("✓ Elasticity features added")
    return df


# ─────────────────────────────────────────────
# 6. ENCODE CATEGORICALS
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode category, season, stock_tier."""

    df = pd.get_dummies(df, columns=["category", "season", "stock_tier"], drop_first=False)

    # Convert boolean dummy columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print("✓ Categorical encoding done")
    return df


# ─────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────
def build_features(filepath: str = "data/raw/synthetic_pricing_data.csv") -> pd.DataFrame:
    df = load_raw_data(filepath)

    df = add_demand_features(df)
    df = add_inventory_features(df)
    df = add_competitor_features(df)
    df = add_temporal_features(df)
    df = add_elasticity_features(df)
    df = encode_categoricals(df)

    # Drop columns not needed for modelling
    df.drop(columns=["timestamp", "product_id"], inplace=True)

    # Drop any remaining nulls (from rolling window first rows)
    before = len(df)
    df.dropna(inplace=True)
    print(f"✓ Dropped {before - len(df)} rows with NaN (rolling window warmup)")

    print(f"\nFinal feature set: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = build_features()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    print("Saved to data/processed/features.csv")

    print("\nFeature columns:")
    for col in sorted(df.columns):
        print(f"  {col}")