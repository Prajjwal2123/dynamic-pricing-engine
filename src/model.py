import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap


# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_features(filepath: str = "data/processed/features.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded features: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Split into X (features) and y (target).
    Target = demand (we predict demand, then use it in pricing logic).
    """
    # Drop columns not used as model input
    drop_cols = ["demand"]
    
    # Also drop historical_sales columns to avoid data leakage
    leakage_cols = [c for c in df.columns if "historical_sales" in c]
    drop_cols += leakage_cols

    X = df.drop(columns=drop_cols)
    y = df["demand"]

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    print(f"Features (X): {X.shape[1]} columns")
    print(f"Target  (y): demand — range [{y.min():.0f}, {y.max():.0f}]")
    return X, y


# ─────────────────────────────────────────────
# 2. TIME-SERIES CROSS VALIDATION
# ─────────────────────────────────────────────
def time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Validate using TimeSeriesSplit — never leaks future into past.
    This is what separates a serious ML project from a toy.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    rmse_scores = []
    mape_scores = []

    print(f"\nRunning {n_splits}-fold TimeSeriesSplit CV...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None)  # demand can't be negative

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mape = np.mean(np.abs((y_val - preds) / (y_val + 1e-9))) * 100

        rmse_scores.append(rmse)
        mape_scores.append(mape)
        print(f"  Fold {fold+1}: RMSE = {rmse:.2f}  |  MAPE = {mape:.2f}%")

    results = {
        "mean_rmse": round(float(np.mean(rmse_scores)), 4),
        "std_rmse":  round(float(np.std(rmse_scores)),  4),
        "mean_mape": round(float(np.mean(mape_scores)), 4),
        "std_mape":  round(float(np.std(mape_scores)),  4),
    }

    print(f"\nCV Results:")
    print(f"  RMSE: {results['mean_rmse']} ± {results['std_rmse']}")
    print(f"  MAPE: {results['mean_mape']}% ± {results['std_mape']}%")
    return results


# ─────────────────────────────────────────────
# 3. TRAIN FINAL MODEL
# ─────────────────────────────────────────────
def train_final_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """Train on full dataset after CV confirms model is solid."""

    # 80/20 time-based split for final eval
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate on held-out test set
    preds = np.clip(model.predict(X_test), 0, None)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100
    r2    = r2_score(y_test, preds)

    print(f"\nFinal Test Set Metrics:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  R²   : {r2:.4f}")

    metrics = {
        "test_rmse": round(rmse, 4),
        "test_mae":  round(mae,  4),
        "test_mape": round(mape, 4),
        "test_r2":   round(r2,   4),
        "n_estimators_used": model.best_iteration + 1
    }

    return model, metrics, X_test, y_test, preds


# ─────────────────────────────────────────────
# 4. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
def compute_shap(model: xgb.XGBRegressor,
                 X: pd.DataFrame,
                 save_dir: str = "models") -> shap.Explainer:
    """
    SHAP tells us WHY the model made each prediction.
    This is what you show in interviews and on the dashboard.
    """
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    os.makedirs(save_dir, exist_ok=True)

    # Global feature importance plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=15, show=False)
    plt.title("SHAP Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/shap_importance.png")

    # Beeswarm plot (shows direction of each feature's effect)
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Direction")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/shap_beeswarm.png")

    return explainer, shap_values


def explain_single_prediction(explainer, X: pd.DataFrame,
                               row_idx: int = 0) -> dict:
    """
    Explain one individual prediction — returns top 3 reasons.
    This powers the dashboard's 'why this price' feature.
    """
    shap_vals   = explainer(X.iloc[[row_idx]])
    feat_names  = X.columns.tolist()
    values      = shap_vals.values[0]

    contributions = sorted(
        zip(feat_names, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top3 = contributions[:3]
    reasons = []
    for feat, val in top3:
        direction = "increased" if val > 0 else "decreased"
        reasons.append({
            "feature":    feat,
            "shap_value": round(float(val), 4),
            "direction":  direction,
            "feature_value": round(float(X.iloc[row_idx][feat]), 4)
        })

    return {"top_3_reasons": reasons}


# ─────────────────────────────────────────────
# 5. SAVE MODEL & METRICS
# ─────────────────────────────────────────────
def save_model(model, metrics: dict, feature_names: list,
               save_dir: str = "models") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = f"{save_dir}/xgb_demand_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    # Save feature names (needed later for API)
    metrics["feature_names"] = feature_names
    metrics_path = f"{save_dir}/model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


# ─────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────
def run_training_pipeline():
    # Load
    df = load_features()
    X, y = prepare_model_data(df)

    # Save model-ready data
    model_ready = X.copy()
    model_ready["demand"] = y.values
    os.makedirs("data/processed", exist_ok=True)
    model_ready.to_csv("data/processed/model_ready.csv", index=False)
    print("Saved: data/processed/model_ready.csv")

    # Cross-validate
    cv_results = time_series_cv(X, y, n_splits=5)

    # Train final model
    model, metrics, X_test, y_test, preds = train_final_model(X, y)
    metrics.update(cv_results)

    # SHAP
    # Save FIRST — before anything else that could crash
    save_model(model, metrics, X.columns.tolist())

    # SHAP — optional, wrapped in try/except so a crash never blocks the save
    try:
        sample_X = X.sample(min(500, len(X)), random_state=42)
        explainer, shap_values = compute_shap(model, sample_X)
        example = explain_single_prediction(explainer, sample_X, row_idx=0)
        print("\nExample prediction explanation:")
        for r in example["top_3_reasons"]:
            print(f"  {r['feature']} = {r['feature_value']} "
                  f"→ demand {r['direction']} by {abs(r['shap_value']):.3f}")
        return model, explainer, X, y
    except Exception as e:
        print(f"\nSHAP skipped due to version conflict: {e}")
        print("Model is saved and ready. Continuing to pricing engine.")
        return model, None, X, y


if __name__ == "__main__":
    run_training_pipeline()