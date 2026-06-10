import numpy as np
import pandas as pd
import joblib
import json
import os
from scipy.optimize import minimize_scalar
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────
def load_model(model_path: str = "models/xgb_demand_model.pkl",
               metrics_path: str = "models/model_metrics.json"):
    model = joblib.load(model_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    feature_names = metrics["feature_names"]
    print(f"Model loaded. Features expected: {len(feature_names)}")
    return model, feature_names


# ─────────────────────────────────────────────
# 2. BUSINESS CONSTRAINTS
# ─────────────────────────────────────────────
class PricingConstraints:
    """
    Hard business rules the ML model cannot violate.
    These protect margin and brand positioning.
    """
    MARGIN_FLOOR         = 0.10   # minimum 10% above cost
    MAX_PREMIUM_OVER_COMP= 0.25   # never more than 25% above competitor
    MAX_DISCOUNT_VS_COMP = 0.15   # never more than 15% below competitor
    CRITICAL_STOCK_BUMP  = 0.15   # bump price 15% when critically low stock
    LOW_STOCK_BUMP       = 0.08   # bump price 8% when low stock
    HIGH_DEMAND_BUMP     = 0.10   # bump price 10% when demand momentum is high
    ELASTIC_CAP          = 0.05   # cap price increase at 5% for elastic products

    @staticmethod
    def compute_bounds(base_price: float,
                       cost: float,
                       competitor_price: float) -> tuple:
        floor = max(cost * (1 + PricingConstraints.MARGIN_FLOOR),
                    competitor_price * (1 - PricingConstraints.MAX_DISCOUNT_VS_COMP))
        ceiling = competitor_price * (1 + PricingConstraints.MAX_PREMIUM_OVER_COMP)
        return round(floor, 2), round(ceiling, 2)


# ─────────────────────────────────────────────
# 3. DEMAND PREDICTION
# ─────────────────────────────────────────────
def predict_demand(model,
                   feature_names: list,
                   input_features: dict,
                   price: float) -> float:
    """
    Predict demand at a given price point.
    Called repeatedly during optimization to find revenue-maximizing price.
    """
    features = {k: input_features.get(k, 0) for k in feature_names}
    features["base_price"] = price

    row = pd.DataFrame([features])[feature_names]
    demand = float(model.predict(row)[0])
    return max(demand, 0)  # demand can't be negative


# ─────────────────────────────────────────────
# 4. REVENUE OPTIMIZATION
# ─────────────────────────────────────────────
def optimize_price(model,
                   feature_names: list,
                   input_features: dict,
                   price_floor: float,
                   price_ceiling: float) -> dict:
    """
    Find the price that maximizes Revenue = price × predicted_demand.
    Uses scipy's minimize_scalar (Brent method) — fast and reliable for 1D.

    We minimize negative revenue (scipy only minimizes, not maximizes).
    """
    def negative_revenue(price):
        demand = predict_demand(model, feature_names, input_features, price)
        return -(price * demand)   # negative because we minimize

    result = minimize_scalar(
        negative_revenue,
        bounds=(price_floor, price_ceiling),
        method="bounded"
    )

    optimal_price    = round(result.x, 2)
    predicted_demand = round(predict_demand(
        model, feature_names, input_features, optimal_price), 2)
    expected_revenue = round(optimal_price * predicted_demand, 2)

    return {
        "optimal_price":     optimal_price,
        "predicted_demand":  predicted_demand,
        "expected_revenue":  expected_revenue,
        "optimization_success": result.success
    }


# ─────────────────────────────────────────────
# 5. RULE-BASED ADJUSTMENT LAYER
# ─────────────────────────────────────────────
def apply_business_rules(optimized_price: float,
                          base_price: float,
                          inventory_ratio: float,
                          demand_momentum: float,
                          price_elasticity: float,
                          price_floor: float,
                          price_ceiling: float) -> tuple:
    """
    Applies human-interpretable business rules on top of the ML output.
    Returns adjusted price + list of reasons explaining the adjustment.
    """
    price   = optimized_price
    reasons = []

    # Rule 1 — Critical stock: aggressive price bump
    if inventory_ratio < 0.10:
        bump   = price * PricingConstraints.CRITICAL_STOCK_BUMP
        price += bump
        reasons.append({
            "rule":   "Critical stock level",
            "effect": f"+{PricingConstraints.CRITICAL_STOCK_BUMP*100:.0f}% (₹{bump:.2f})",
            "why":    "Inventory < 10% — scarcity pricing applied"
        })

    # Rule 2 — Low stock: moderate bump
    elif inventory_ratio < 0.25:
        bump   = price * PricingConstraints.LOW_STOCK_BUMP
        price += bump
        reasons.append({
            "rule":   "Low stock level",
            "effect": f"+{PricingConstraints.LOW_STOCK_BUMP*100:.0f}% (₹{bump:.2f})",
            "why":    "Inventory < 25% — moderate scarcity pricing"
        })

    # Rule 3 — High demand momentum: capitalize on trend
    if demand_momentum > 10:
        bump   = price * PricingConstraints.HIGH_DEMAND_BUMP
        price += bump
        reasons.append({
            "rule":   "High demand momentum",
            "effect": f"+{PricingConstraints.HIGH_DEMAND_BUMP*100:.0f}% (₹{bump:.2f})",
            "why":    f"7d demand trending {demand_momentum:.1f} units above 30d avg"
        })

    # Rule 4 — Elastic product: cap any price increase
    if price_elasticity < -1.0 and price > base_price:
        max_increase = base_price * (1 + PricingConstraints.ELASTIC_CAP)
        if price > max_increase:
            price = max_increase
            reasons.append({
                "rule":   "Elastic product — price increase capped",
                "effect": f"Capped at +{PricingConstraints.ELASTIC_CAP*100:.0f}%",
                "why":    f"Elasticity = {price_elasticity:.2f} — aggressive price hike would kill demand"
            })

    # Enforce hard bounds (floor and ceiling always win)
    price = float(np.clip(price, price_floor, price_ceiling))

    if not reasons:
        reasons.append({
            "rule":   "ML optimization",
            "effect": "No rule overrides triggered",
            "why":    "Price set purely by revenue maximization"
        })

    return round(price, 2), reasons


# ─────────────────────────────────────────────
# 6. FULL RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def recommend_price(model,
                    feature_names: list,
                    product_id: str,
                    base_price: float,
                    cost: float,
                    competitor_price: float,
                    inventory_ratio: float,
                    demand_momentum: float,
                    price_elasticity: float,
                    extra_features: dict = None) -> dict:
    """
    Master function — takes product context, returns a full pricing recommendation.
    This is what the API and dashboard will call.
    """
    # Build feature dict for the model
    input_features = {
        "base_price":        base_price,
        "competitor_price":  competitor_price,
        "inventory_ratio":   inventory_ratio,
        "demand_momentum":   demand_momentum,
        "price_elasticity":  price_elasticity,
        "price_gap":         base_price - competitor_price,
        "price_gap_pct":     ((base_price - competitor_price) / competitor_price) * 100,
        "is_cheaper":        int(base_price < competitor_price),
        "is_low_stock":      int(inventory_ratio < 0.25),
        "is_critical_stock": int(inventory_ratio < 0.10),
    }
    if extra_features:
        input_features.update(extra_features)

    # Compute price bounds
    price_floor, price_ceiling = PricingConstraints.compute_bounds(
        base_price, cost, competitor_price
    )

    # ML optimization
    opt = optimize_price(model, feature_names, input_features,
                         price_floor, price_ceiling)

    # Business rule layer
    final_price, reasons = apply_business_rules(
        optimized_price  = opt["optimal_price"],
        base_price       = base_price,
        inventory_ratio  = inventory_ratio,
        demand_momentum  = demand_momentum,
        price_elasticity = price_elasticity,
        price_floor      = price_floor,
        price_ceiling    = price_ceiling
    )

    # Revenue impact vs current price
    current_revenue  = base_price * opt["predicted_demand"]
    new_revenue      = final_price * opt["predicted_demand"]
    revenue_delta    = round(new_revenue - current_revenue, 2)
    revenue_delta_pct= round((revenue_delta / (current_revenue + 1e-9)) * 100, 2)

    recommendation = {
        "product_id":          product_id,
        "timestamp":           datetime.now().isoformat(),
        "current_price":       base_price,
        "recommended_price":   final_price,
        "price_change":        round(final_price - base_price, 2),
        "price_change_pct":    round(((final_price - base_price) / base_price) * 100, 2),
        "price_floor":         price_floor,
        "price_ceiling":       price_ceiling,
        "predicted_demand":    opt["predicted_demand"],
        "expected_revenue":    round(new_revenue, 2),
        "revenue_delta":       revenue_delta,
        "revenue_delta_pct":   revenue_delta_pct,
        "competitor_price":    competitor_price,
        "inventory_ratio":     inventory_ratio,
        "reasoning":           reasons,
    }

    return recommendation


# ─────────────────────────────────────────────
# 7. BATCH PRICING + LOGGING
# ─────────────────────────────────────────────
def run_batch_pricing(model,
                      feature_names: list,
                      products: list,
                      log_path: str = "data/processed/pricing_logs.csv") -> pd.DataFrame:
    """
    Run pricing recommendations for a list of products.
    Logs every recommendation — important for audit trails in production.
    """
    results = []
    for p in products:
        try:
            rec = recommend_price(
                model           = model,
                feature_names   = feature_names,
                product_id      = p["product_id"],
                base_price      = p["base_price"],
                cost            = p["cost"],
                competitor_price= p["competitor_price"],
                inventory_ratio = p["inventory_ratio"],
                demand_momentum = p.get("demand_momentum", 0),
                price_elasticity= p.get("price_elasticity", -1),
                extra_features  = p.get("extra_features", {})
            )
            results.append(rec)
            print(f"  {rec['product_id']:10s} | "
                  f"Current: ₹{rec['current_price']:8.2f} | "
                  f"Recommended: ₹{rec['recommended_price']:8.2f} | "
                  f"Revenue Δ: ₹{rec['revenue_delta']:+.2f} ({rec['revenue_delta_pct']:+.1f}%)")
        except Exception as e:
            print(f"  ERROR for {p.get('product_id','?')}: {e}")

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

    print(f"\nLogged {len(df)} recommendations → {log_path}")
    return df


# ─────────────────────────────────────────────
# DEMO RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, feature_names = load_model()

    # Sample products to price
    sample_products = [
        {
            "product_id":       "P0001",
            "base_price":       1200.00,
            "cost":             800.00,
            "competitor_price": 1150.00,
            "inventory_ratio":  0.08,      # critical stock!
            "demand_momentum":  15.0,      # demand rising
            "price_elasticity": -0.8,      # inelastic
        },
        {
            "product_id":       "P0002",
            "base_price":       500.00,
            "cost":             300.00,
            "competitor_price": 480.00,
            "inventory_ratio":  0.60,      # healthy stock
            "demand_momentum":  -5.0,      # demand falling
            "price_elasticity": -1.8,      # elastic
        },
        {
            "product_id":       "P0003",
            "base_price":       250.00,
            "cost":             150.00,
            "competitor_price": 280.00,    # competitor is more expensive
            "inventory_ratio":  0.20,      # low stock
            "demand_momentum":  2.0,       # stable demand
            "price_elasticity": -1.1,      # slightly elastic
        },
        {
            "product_id":       "P0004",
            "base_price":       3500.00,
            "cost":             2200.00,
            "competitor_price": 3600.00,
            "inventory_ratio":  0.45,
            "demand_momentum":  8.0,
            "price_elasticity": -0.5,      # inelastic (premium product)
        },
    ]

    print("\n" + "="*70)
    print("DYNAMIC PRICING ENGINE — BATCH RECOMMENDATIONS")
    print("="*70 + "\n")

    results_df = run_batch_pricing(model, feature_names, sample_products)

    print("\n\nDETAILED REASONING FOR EACH PRODUCT:")
    print("="*70)
    for _, row in results_df.iterrows():
        print(f"\nProduct: {row['product_id']}")
        print(f"  Price: ₹{row['current_price']} → ₹{row['recommended_price']} "
              f"({row['price_change_pct']:+.1f}%)")
        print(f"  Revenue impact: ₹{row['revenue_delta']:+.2f} "
              f"({row['revenue_delta_pct']:+.1f}%)")
        print(f"  Reasoning:")
        for r in row["reasoning"]:
            print(f"    • {r['rule']}: {r['why']}")