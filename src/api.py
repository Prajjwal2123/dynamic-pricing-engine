import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import uvicorn

from pricing_engine import recommend_price, run_batch_pricing

# ─────────────────────────────────────────────
# BASE DIRECTORY — absolute path to project root
# Works regardless of where you run the script from
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def abs_path(*parts):
    """Helper to build absolute paths from project root."""
    return os.path.join(BASE_DIR, *parts)


# ─────────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Dynamic Pricing Engine API",
    description = "ML-powered pricing recommendations using XGBoost + demand forecasting",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
# 2. LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
MODEL         = None
FEATURE_NAMES = None

@app.on_event("startup")
def load_model():
    global MODEL, FEATURE_NAMES
    try:
        model_path   = abs_path("models", "xgb_demand_model.pkl")
        metrics_path = abs_path("models", "model_metrics.json")

        print(f"Loading model from: {model_path}")

        MODEL = joblib.load(model_path)

        with open(metrics_path) as f:
            metrics = json.load(f)

        FEATURE_NAMES = metrics["feature_names"]
        print(f"Model loaded successfully. Features: {len(FEATURE_NAMES)}")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"Make sure you ran: python src/model.py first")


# ─────────────────────────────────────────────
# 3. REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class PricingRequest(BaseModel):
    product_id:       str   = Field(...,  example="P0001")
    base_price:       float = Field(...,  gt=0, example=1200.0)
    cost:             float = Field(...,  gt=0, example=800.0)
    competitor_price: float = Field(...,  gt=0, example=1150.0)
    inventory_ratio:  float = Field(...,  ge=0, le=1, example=0.08)
    demand_momentum:  float = Field(0.0,  example=15.0)
    price_elasticity: float = Field(-1.0, example=-0.8)

    @validator("base_price")
    def price_must_cover_cost(cls, v, values):
        if "cost" in values and v < values["cost"]:
            raise ValueError("base_price must be greater than cost")
        return v

    @validator("inventory_ratio")
    def inventory_must_be_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("inventory_ratio must be between 0 and 1")
        return v

    class Config:
        schema_extra = {
            "example": {
                "product_id":       "P0001",
                "base_price":       1200.0,
                "cost":             800.0,
                "competitor_price": 1150.0,
                "inventory_ratio":  0.08,
                "demand_momentum":  15.0,
                "price_elasticity": -0.8
            }
        }


class BatchPricingRequest(BaseModel):
    products: List[PricingRequest] = Field(..., min_items=1, max_items=100)


class PricingResponse(BaseModel):
    product_id:        str
    timestamp:         str
    current_price:     float
    recommended_price: float
    price_change:      float
    price_change_pct:  float
    price_floor:       float
    price_ceiling:     float
    predicted_demand:  float
    expected_revenue:  float
    revenue_delta:     float
    revenue_delta_pct: float
    competitor_price:  float
    inventory_ratio:   float
    reasoning:         list


# ─────────────────────────────────────────────
# 4. ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "Dynamic Pricing Engine",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health")
def health():
    """Health check — confirms model loaded correctly."""
    return {
        "status":         "healthy" if MODEL is not None else "model not loaded",
        "model_loaded":   MODEL is not None,
        "features_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "timestamp":      datetime.now().isoformat()
    }


@app.get("/model-info")
def model_info():
    """Returns model performance metrics."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        metrics_path = abs_path("models", "model_metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        return {
            "model_type":   "XGBoost Regressor",
            "target":       "demand",
            "test_rmse":    metrics.get("test_rmse"),
            "test_mape":    metrics.get("test_mape"),
            "test_r2":      metrics.get("test_r2"),
            "cv_mean_mape": metrics.get("mean_mape"),
            "n_features":   len(FEATURE_NAMES),
            "n_estimators": metrics.get("n_estimators_used"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-price", response_model=PricingResponse)
def recommend_price_endpoint(request: PricingRequest):
    """
    Core endpoint — takes product context, returns optimal price recommendation.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = recommend_price(
            model            = MODEL,
            feature_names    = FEATURE_NAMES,
            product_id       = request.product_id,
            base_price       = request.base_price,
            cost             = request.cost,
            competitor_price = request.competitor_price,
            inventory_ratio  = request.inventory_ratio,
            demand_momentum  = request.demand_momentum,
            price_elasticity = request.price_elasticity,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pricing error: {str(e)}")


@app.post("/batch-recommend")
def batch_recommend_endpoint(request: BatchPricingRequest):
    """
    Batch endpoint — price multiple products in one API call.
    Used for nightly pricing runs across an entire catalogue.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        products   = [p.dict() for p in request.products]
        log_path   = abs_path("data", "processed", "pricing_logs.csv")
        results_df = run_batch_pricing(MODEL, FEATURE_NAMES, products,
                                       log_path=log_path)
        results = results_df.to_dict(orient="records")
        return {
            "total_products":      len(results),
            "total_revenue_delta": round(
                sum(r.get("revenue_delta", 0) for r in results), 2),
            "recommendations":     results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


@app.get("/price-curve/{product_id}")
def price_curve(
    product_id:       str,
    base_price:       float,
    cost:             float,
    competitor_price: float,
    inventory_ratio:  float = 0.5,
    demand_momentum:  float = 0.0,
    price_elasticity: float = -1.0,
    steps:            int   = 20
):
    """
    Returns revenue and demand at different price points.
    Used by the dashboard to draw the price vs revenue curve.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        from pricing_engine import PricingConstraints, predict_demand

        price_floor, price_ceiling = PricingConstraints.compute_bounds(
            base_price, cost, competitor_price
        )

        input_features = {
            "base_price":        base_price,
            "competitor_price":  competitor_price,
            "inventory_ratio":   inventory_ratio,
            "demand_momentum":   demand_momentum,
            "price_elasticity":  price_elasticity,
            "price_gap":         base_price - competitor_price,
            "price_gap_pct":     ((base_price - competitor_price)
                                  / competitor_price) * 100,
            "is_cheaper":        int(base_price < competitor_price),
            "is_low_stock":      int(inventory_ratio < 0.25),
            "is_critical_stock": int(inventory_ratio < 0.10),
        }

        curve = []
        for p in np.linspace(price_floor, price_ceiling, steps):
            d = predict_demand(MODEL, FEATURE_NAMES, input_features, p)
            curve.append({
                "price":   round(float(p), 2),
                "demand":  round(float(d), 2),
                "revenue": round(float(p * d), 2)
            })

        return {
            "product_id":    product_id,
            "price_floor":   round(price_floor, 2),
            "price_ceiling": round(price_ceiling, 2),
            "curve":         curve
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# 5. RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        reload_dirs = [str(BASE_DIR)]
    )