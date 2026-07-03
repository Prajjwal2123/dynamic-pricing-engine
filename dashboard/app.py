import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
import os
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "Dynamic Pricing Engine",
    page_icon  = "💰",
    layout     = "wide"
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def color_delta(val):
    return "🟢" if val >= 0 else "🔴"

def fmt_inr(val):
    return f"₹{val:,.2f}"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Pricing Engine")
    st.markdown("---")

    # API health check
    health = api_get("/health")
    if health and health.get("model_loaded"):
        st.success(f"API Online ✓\nFeatures: {health['features_count']}")
    else:
        st.error("API Offline — run: python src/api.py")
        st.stop()

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠 Single Product",
        "📦 Batch Pricing",
        "📈 Price vs Revenue Curve",
        "📋 Pricing Logs",
        "🤖 Model Info"
    ])

    st.markdown("---")
    # Model metrics in sidebar
    info = api_get("/model-info")
    if info:
        st.markdown("### Model Performance")
        st.metric("MAPE",  f"{info['test_mape']}%")
        st.metric("R²",    f"{info['test_r2']}")
        st.metric("RMSE",  f"{info['test_rmse']}")


# ─────────────────────────────────────────────
# PAGE 1 — SINGLE PRODUCT RECOMMENDATION
# ─────────────────────────────────────────────
if page == "🏠 Single Product":
    st.title("Single Product Pricing Recommendation")
    st.markdown("Enter product details to get an ML-powered price recommendation.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Product Details")
        product_id       = st.text_input("Product ID", value="P0001")
        base_price       = st.number_input("Current Price (₹)", min_value=1.0,
                                            value=1200.0, step=10.0)
        cost             = st.number_input("Cost Price (₹)", min_value=1.0,
                                            value=800.0, step=10.0)
        competitor_price = st.number_input("Competitor Price (₹)", min_value=1.0,
                                            value=1150.0, step=10.0)

    with col2:
        st.markdown("### Market Signals")
        inventory_ratio  = st.slider("Inventory Level",
                                      min_value=0.0, max_value=1.0,
                                      value=0.08, step=0.01,
                                      help="0 = empty, 1 = fully stocked")
        demand_momentum  = st.slider("Demand Momentum (7d vs 30d avg)",
                                      min_value=-50.0, max_value=50.0,
                                      value=15.0, step=1.0,
                                      help="Positive = demand rising")
        price_elasticity = st.slider("Price Elasticity",
                                      min_value=-5.0, max_value=0.0,
                                      value=-0.8, step=0.1,
                                      help="< -1 = elastic, > -1 = inelastic")

    # Inventory status indicator
    if inventory_ratio < 0.10:
        st.warning(f"⚠️ Critical stock level ({inventory_ratio*100:.0f}%) — scarcity pricing will apply")
    elif inventory_ratio < 0.25:
        st.info(f"📦 Low stock ({inventory_ratio*100:.0f}%) — moderate price bump expected")
    else:
        st.success(f"✅ Healthy stock ({inventory_ratio*100:.0f}%)")

    st.markdown("---")

    if st.button("🎯 Get Price Recommendation", type="primary", use_container_width=True):
        if base_price <= cost:
            st.error("Current price must be greater than cost price.")
        else:
            with st.spinner("Calling pricing engine..."):
                payload = {
                    "product_id":       product_id,
                    "base_price":       base_price,
                    "cost":             cost,
                    "competitor_price": competitor_price,
                    "inventory_ratio":  inventory_ratio,
                    "demand_momentum":  demand_momentum,
                    "price_elasticity": price_elasticity
                }
                result = api_post("/recommend-price", payload)

            if result:
                st.markdown("---")
                st.markdown("### Recommendation")

                # Key metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price",      fmt_inr(result["current_price"]))
                m2.metric("Recommended Price",  fmt_inr(result["recommended_price"]),
                          delta=f"{result['price_change_pct']:+.1f}%")
                m3.metric("Predicted Demand",   f"{result['predicted_demand']:.0f} units")
                m4.metric("Revenue Impact",     fmt_inr(result["revenue_delta"]),
                          delta=f"{result['revenue_delta_pct']:+.1f}%")

                st.markdown("---")
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### Price Boundaries")
                    fig, ax = plt.subplots(figsize=(7, 2))
                    ax.barh(["Range"], [result["price_ceiling"] - result["price_floor"]],
                            left=[result["price_floor"]], color="lightblue",
                            height=0.4, label="Allowed range")
                    ax.scatter(result["current_price"],     0,
                               color="steelblue", s=150, zorder=5,
                               label=f"Current ₹{result['current_price']:.0f}")
                    ax.scatter(result["recommended_price"], 0,
                               color="green", s=200, marker="D", zorder=5,
                               label=f"Recommended ₹{result['recommended_price']:.0f}")
                    ax.scatter(result["competitor_price"],  0,
                               color="orange", s=150, marker="^", zorder=5,
                               label=f"Competitor ₹{result['competitor_price']:.0f}")
                    ax.set_xlabel("Price (₹)")
                    ax.set_yticks([])
                    ax.legend(loc="upper right", fontsize=8)
                    ax.set_title("Price Positioning")
                    st.pyplot(fig)
                    plt.close()

                with col_b:
                    st.markdown("### Why This Price?")
                    for r in result["reasoning"]:
                        with st.expander(f"📌 {r['rule']} — {r['effect']}", expanded=True):
                            st.write(r["why"])

                # Revenue comparison
                st.markdown("### Revenue Comparison")
                current_rev = result["current_price"] * result["predicted_demand"]
                new_rev     = result["recommended_price"] * result["predicted_demand"]

                fig2, ax2 = plt.subplots(figsize=(6, 3))
                bars = ax2.bar(["Current Price", "Recommended Price"],
                               [current_rev, new_rev],
                               color=["steelblue", "green"], alpha=0.85, width=0.4)
                ax2.set_ylabel("Expected Revenue (₹)")
                ax2.set_title("Revenue Impact")
                for bar, val in zip(bars, [current_rev, new_rev]):
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 500,
                             fmt_inr(val), ha="center", fontsize=10, fontweight="bold")
                st.pyplot(fig2)
                plt.close()

            else:
                st.error("API call failed. Make sure the API server is running.")


# ─────────────────────────────────────────────
# PAGE 2 — BATCH PRICING
# ─────────────────────────────────────────────
elif page == "📦 Batch Pricing":
    st.title("Batch Pricing — Multiple Products")
    st.markdown("Price your entire catalogue at once.")

    st.markdown("### Sample Products (editable)")

    default_data = pd.DataFrame([
        {"product_id":"P0001","base_price":1200,"cost":800,
         "competitor_price":1150,"inventory_ratio":0.08,
         "demand_momentum":15.0,"price_elasticity":-0.8},
        {"product_id":"P0002","base_price":500,"cost":300,
         "competitor_price":480,"inventory_ratio":0.60,
         "demand_momentum":-5.0,"price_elasticity":-1.8},
        {"product_id":"P0003","base_price":250,"cost":150,
         "competitor_price":280,"inventory_ratio":0.20,
         "demand_momentum":2.0,"price_elasticity":-1.1},
        {"product_id":"P0004","base_price":3500,"cost":2200,
         "competitor_price":3600,"inventory_ratio":0.45,
         "demand_momentum":8.0,"price_elasticity":-0.5},
    ])

    edited_df = st.data_editor(default_data, use_container_width=True,
                                num_rows="dynamic")

    if st.button("🚀 Run Batch Pricing", type="primary", use_container_width=True):
        products = edited_df.to_dict(orient="records")
        with st.spinner(f"Pricing {len(products)} products..."):
            result = api_post("/batch-recommend", {"products": products})

        if result:
            st.success(f"✅ {result['total_products']} products priced | "
                       f"Total revenue delta: {fmt_inr(result['total_revenue_delta'])}")

            recs = pd.DataFrame(result["recommendations"])
            display_cols = ["product_id", "current_price", "recommended_price",
                            "price_change_pct", "predicted_demand",
                            "revenue_delta", "revenue_delta_pct"]
            st.dataframe(recs[display_cols].style.format({
                "current_price":     "₹{:.2f}",
                "recommended_price": "₹{:.2f}",
                "price_change_pct":  "{:+.1f}%",
                "predicted_demand":  "{:.0f}",
                "revenue_delta":     "₹{:+.2f}",
                "revenue_delta_pct": "{:+.1f}%",
            }), use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            x     = range(len(recs))
            width = 0.35
            ax.bar([i - width/2 for i in x], recs["current_price"],
                   width, label="Current",     color="steelblue", alpha=0.8)
            ax.bar([i + width/2 for i in x], recs["recommended_price"],
                   width, label="Recommended", color="green",     alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(recs["product_id"])
            ax.set_ylabel("Price (₹)")
            ax.set_title("Current vs Recommended Prices")
            ax.legend()
            st.pyplot(fig)
            plt.close()
        else:
            st.error("Batch pricing failed.")


# ─────────────────────────────────────────────
# PAGE 3 — PRICE VS REVENUE CURVE
# ─────────────────────────────────────────────
elif page == "📈 Price vs Revenue Curve":
    st.title("Price vs Revenue Optimization Curve")
    st.markdown("See how revenue and demand change across the full price range.")

    col1, col2 = st.columns(2)
    with col1:
        pid   = st.text_input("Product ID", value="P0001")
        bp    = st.number_input("Current Price (₹)", value=1200.0, step=10.0)
        cost  = st.number_input("Cost (₹)",          value=800.0,  step=10.0)
    with col2:
        comp  = st.number_input("Competitor Price (₹)", value=1150.0, step=10.0)
        inv   = st.slider("Inventory Ratio", 0.0, 1.0, 0.08, 0.01)
        elast = st.slider("Price Elasticity", -5.0, 0.0, -0.8, 0.1)

    if st.button("📊 Generate Curve", type="primary", use_container_width=True):
        params = {
            "base_price": bp, "cost": cost,
            "competitor_price": comp,
            "inventory_ratio": inv,
            "price_elasticity": elast,
            "steps": 30
        }
        with st.spinner("Computing price-revenue curve..."):
            data = api_get(f"/price-curve/{pid}?" +
                           "&".join(f"{k}={v}" for k, v in params.items()))

        if data:
            curve_df = pd.DataFrame(data["curve"])
            best_row = curve_df.loc[curve_df["revenue"].idxmax()]

            st.success(f"Optimal price: ₹{best_row['price']:.2f} → "
                       f"Revenue: ₹{best_row['revenue']:,.2f} | "
                       f"Demand: {best_row['demand']:.0f} units")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Revenue curve
            axes[0].plot(curve_df["price"], curve_df["revenue"],
                         color="steelblue", linewidth=2.5)
            axes[0].fill_between(curve_df["price"], curve_df["revenue"],
                                  alpha=0.1, color="steelblue")
            axes[0].axvline(bp,                  color="gray",
                            linestyle="--", label=f"Current ₹{bp:.0f}")
            axes[0].axvline(best_row["price"],   color="green",
                            linestyle="-",  linewidth=2,
                            label=f"Optimal ₹{best_row['price']:.0f}")
            axes[0].axvline(data["price_floor"],   color="red",
                            linestyle=":", alpha=0.5, label="Floor")
            axes[0].axvline(data["price_ceiling"], color="red",
                            linestyle=":", alpha=0.5, label="Ceiling")
            axes[0].set_xlabel("Price (₹)")
            axes[0].set_ylabel("Expected Revenue (₹)")
            axes[0].set_title("Revenue Optimization Curve")
            axes[0].legend(fontsize=9)

            # Demand curve
            axes[1].plot(curve_df["price"], curve_df["demand"],
                         color="coral", linewidth=2.5)
            axes[1].fill_between(curve_df["price"], curve_df["demand"],
                                  alpha=0.1, color="coral")
            axes[1].axvline(bp,                color="gray",
                            linestyle="--", label=f"Current ₹{bp:.0f}")
            axes[1].axvline(best_row["price"], color="green",
                            linestyle="-",  linewidth=2,
                            label=f"Optimal ₹{best_row['price']:.0f}")
            axes[1].set_xlabel("Price (₹)")
            axes[1].set_ylabel("Predicted Demand (units)")
            axes[1].set_title("Demand Curve")
            axes[1].legend(fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Full Price-Demand-Revenue Table")
            st.dataframe(curve_df.style.format({
                "price":   "₹{:.2f}",
                "demand":  "{:.1f}",
                "revenue": "₹{:,.2f}"
            }), use_container_width=True)
        else:
            st.error("Failed to fetch curve data.")


# ─────────────────────────────────────────────
# PAGE 4 — PRICING LOGS
# ─────────────────────────────────────────────
elif page == "📋 Pricing Logs":
    st.title("Pricing Decision Log")
    st.markdown("Every recommendation made by the engine — full audit trail.")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(BASE_DIR, "data", "processed", "pricing_logs.csv")

    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)

        st.metric("Total Recommendations", len(logs))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Price Change",
                      f"{logs['price_change_pct'].mean():+.1f}%")
        with col2:
            st.metric("Total Revenue Delta",
                      fmt_inr(logs["revenue_delta"].sum()))

        st.markdown("### All Recommendations")
        display = ["product_id", "timestamp", "current_price",
                   "recommended_price", "price_change_pct",
                   "revenue_delta", "revenue_delta_pct"]
        existing = [c for c in display if c in logs.columns]
        st.dataframe(logs[existing].sort_values("timestamp", ascending=False),
                     use_container_width=True)

        # Revenue delta chart
        if "revenue_delta" in logs.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ["green" if v >= 0 else "red"
                      for v in logs["revenue_delta"]]
            ax.bar(range(len(logs)), logs["revenue_delta"],
                   color=colors, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Recommendation #")
            ax.set_ylabel("Revenue Delta (₹)")
            ax.set_title("Revenue Impact per Recommendation")
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No logs yet. Make some recommendations first.")


# ─────────────────────────────────────────────
# PAGE 5 — MODEL INFO
# ─────────────────────────────────────────────
elif page == "🤖 Model Info":
    st.title("Model Information & Performance")

    info = api_get("/model-info")
    if info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Type", "XGBoost")
        col2.metric("MAPE",       f"{info['test_mape']}%",
                    delta="Under 10% = production grade",
                    delta_color="off")
        col3.metric("R² Score",   f"{info['test_r2']}")
        col4.metric("Trees Used", f"{info['n_estimators']}")

        st.markdown("---")
        st.markdown("### What these metrics mean")

        c1, c2 = st.columns(2)
        with c1:
            st.info(f"""
**MAPE: {info['test_mape']}%**

Mean Absolute Percentage Error.
On average the model's demand prediction
is off by {info['test_mape']}%.
Under 10% is considered production-grade
for demand forecasting.
""")
            st.info(f"""
**R² Score: {info['test_r2']}**

The model explains {info['test_r2']*100:.1f}% of
the variance in demand. Closer to 1.0 = better.
""")
        with c2:
            st.info(f"""
**CV MAPE: {info['cv_mean_mape']}%**

Cross-validation MAPE across 5 time-series
folds. Low variance = model is consistent
and not overfitting to one time period.
""")
            st.info(f"""
**Trees: {info['n_estimators']}**

XGBoost used early stopping — stopped at
{info['n_estimators']} trees instead of 500.
This prevents overfitting automatically.
""")

        st.markdown("---")
        st.markdown("### SHAP Feature Importance")

        BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shap_path  = os.path.join(BASE_DIR, "models", "shap_importance.png")
        bees_path  = os.path.join(BASE_DIR, "models", "shap_beeswarm.png")

        if os.path.exists(shap_path):
            t1, t2 = st.tabs(["Feature Importance", "Beeswarm Plot"])
            with t1:
                st.image(shap_path, caption="SHAP Feature Importance — top 15 features")
                st.markdown("""
**How to read this:** Longer bar = feature has more influence on demand predictions.
Features like `inventory`, `base_price`, and `demand_momentum` being at the top
confirms the model learned real economic signals.
""")
            with t2:
                st.image(bees_path, caption="SHAP Beeswarm — direction of each feature's effect")
                st.markdown("""
**How to read this:** Red = high feature value, Blue = low.
Position left of center = pushes demand down.
Position right of center = pushes demand up.
""")
        else:
            st.warning("SHAP plots not found. Run python src/model.py first.")