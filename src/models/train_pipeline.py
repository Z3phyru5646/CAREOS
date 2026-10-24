"""
CAREOS Module 1 - Two-Stage ML Pipeline
Stage 1: Consultation Duration Predictor (XGBoost/LightGBM)
Stage 2: Wait-Time Predictor with Asymmetric Quad-Quad Loss
+ Conformalized Quantile Regression (CQR) + Adaptive Conformal Inference (ACI)
+ TreeSHAP Explanations
+ LSTM Sequence Encoder
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from scipy.stats import norm
import shap
import joblib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.queue_feature_engine import (
    get_feature_columns, get_stage1_feature_columns,
    get_stage2_feature_columns, compute_queue_features,
)


# ==============================================================
# Custom Asymmetric Quad-Quad Loss for LightGBM (Section 5.4)
# ==============================================================

class AsymmetricQuadQuadObjective:
    """LightGBM custom objective with Asymmetric Quad-Quad Loss."""
    def __init__(self, alpha_asym: float = 0.8):
        self.alpha_asym = alpha_asym

    def __call__(self, y_true, y_pred):
        residual = y_true - y_pred
        grad = np.where(
            residual > 0,
            -4 * self.alpha_asym * residual,
            -4 * (1 - self.alpha_asym) * residual,
        )
        hess = np.where(
            residual > 0,
            4 * self.alpha_asym * np.ones_like(residual),
            4 * (1 - self.alpha_asym) * np.ones_like(residual),
        )
        return grad, hess


class AsymmetricQuadQuadEval:
    """LightGBM custom evaluation metric for Asymmetric Quad-Quad Loss."""
    def __init__(self, alpha_asym: float = 0.8):
        self.alpha_asym = alpha_asym

    def __call__(self, y_true, y_pred):
        residual = y_true - y_pred
        loss = np.where(
            residual > 0,
            2 * self.alpha_asym * residual ** 2,
            2 * (1 - self.alpha_asym) * residual ** 2,
        )
        return "quad_quad_loss", float(np.mean(loss)), False


def asymmetric_quad_quad_loss(alpha_asym: float = 0.8):
    """
    Returns (objective_fn, eval_fn) for LightGBM custom asymmetric loss.
    L(e; alpha) = 2*alpha * e^2       if e = Y - Y_hat > 0 (underestimation)
                  2*(1-alpha) * e^2    if e <= 0 (overestimation)
    """
    return AsymmetricQuadQuadObjective(alpha_asym), AsymmetricQuadQuadEval(alpha_asym)



# ==============================================================
# Dissatisfaction-Aware Asymmetric Error Score (DAES) - Section 5.4
# ==============================================================

def compute_daes(y_true, y_pred, gamma_threshold: float = 10.0, alpha_asym: float = 0.8):
    """
    DAES metric that penalizes underestimation more but also
    penalizes excessive overestimation beyond gamma minutes.
    """
    residual = y_true - y_pred
    under = residual > 0
    over = residual <= 0
    excessive_over = (-residual) > gamma_threshold

    score = np.zeros_like(residual, dtype=float)
    score[under] = alpha_asym * residual[under] ** 2
    score[over & ~excessive_over] = (1 - alpha_asym) * residual[over & ~excessive_over] ** 2
    score[over & excessive_over] = (1 - alpha_asym) * residual[over & excessive_over] ** 2 + \
                                   0.5 * ((-residual[over & excessive_over]) - gamma_threshold) ** 2

    return np.mean(score)


# ==============================================================
# Stage 1: Consultation Duration Predictor
# ==============================================================

class Stage1DurationPredictor:
    """
    Predicts individual patient consultation duration S_hat_j
    from clinical covariates using XGBoost.
    """

    def __init__(self):
        self.model = None
        self.feature_cols = get_stage1_feature_columns()

    def prepare_data(self, df: pd.DataFrame):
        """Prepare Stage-1 training data."""
        mask = (df["status"] == "COMPLETED") & (df["actual_service_duration"].notna())
        train_df = df[mask].copy()

        # Encode categoricals
        if "complaint_category" in train_df.columns:
            train_df["complaint_cat_encoded"] = pd.Categorical(
                train_df["complaint_category"]
            ).codes
            if "complaint_cat_encoded" not in self.feature_cols:
                self.feature_cols.append("complaint_cat_encoded")

        X = train_df[self.feature_cols].values.astype(np.float32)
        y = train_df["actual_service_duration"].values.astype(np.float32)

        return X, y, train_df

    def train(self, df: pd.DataFrame, save_path: str = None):
        """Train Stage-1 model."""
        X, y, train_df = self.prepare_data(df)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        print(f"[Stage-1 Duration Predictor] MAE: {mae:.2f} min, RMSE: {rmse:.2f} min, R²: {r2:.4f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"  Model saved to: {save_path}")

        return {"mae": mae, "rmse": rmse, "r2": r2}

    def predict(self, X):
        """Predict consultation duration."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)


# ==============================================================
# Stage 2: Wait-Time Predictor with CQR + ACI
# ==============================================================

class Stage2WaitTimePredictor:
    """
    Predicts patient wait time using:
    - LightGBM with Asymmetric Quad-Quad Loss (point prediction)
    - LightGBM Quantile Regressors (tau=0.05, 0.95)
    - Conformalized Quantile Regression (CQR)
    - Adaptive Conformal Inference (ACI) for online calibration
    - TreeSHAP explanations
    """

    def __init__(self, alpha_asym: float = 0.8, alpha_target: float = 0.10, aci_gamma: float = 0.03):
        self.alpha_asym = alpha_asym
        self.alpha_target = alpha_target
        self.aci_gamma = aci_gamma

        self.point_model = None
        self.quantile_lo_model = None  # tau = 0.05
        self.quantile_hi_model = None  # tau = 0.95
        self.cqr_Q = None  # Calibrated conformal quantile
        self.aci_alpha_t = alpha_target  # Current ACI alpha
        self.feature_cols = get_stage2_feature_columns()
        self.shap_explainer = None

    def prepare_data(self, df: pd.DataFrame):
        """Prepare Stage-2 training data."""
        mask = (df["status"] == "COMPLETED") & (df["actual_wait_time"].notna())
        train_df = df[mask].copy()

        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0

        X = train_df[self.feature_cols].values.astype(np.float32)
        y = train_df["actual_wait_time"].values.astype(np.float32)

        return X, y, train_df

    def train(self, df: pd.DataFrame, save_dir: str = None):
        """Train all Stage-2 models."""
        X, y, train_df = self.prepare_data(df)

        # Split: 60% train, 20% calibration, 20% test
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

        # ---- 1. Point Model with Asymmetric Quad-Quad Loss ----
        obj_fn, eval_fn = asymmetric_quad_quad_loss(self.alpha_asym)

        self.point_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective=obj_fn,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.point_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=eval_fn,
        )

        # ---- 2. Quantile Regressors ----
        # Lower quantile (tau = 0.05)
        self.quantile_lo_model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            objective="quantile",
            alpha=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.quantile_lo_model.fit(X_train, y_train)

        # Upper quantile (tau = 0.95)
        self.quantile_hi_model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            objective="quantile",
            alpha=0.95,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.quantile_hi_model.fit(X_train, y_train)

        # ---- 3. CQR Calibration ----
        q_lo_cal = self.quantile_lo_model.predict(X_cal)
        q_hi_cal = self.quantile_hi_model.predict(X_cal)

        # Nonconformity scores: E_k = max(q_lo - Y, Y - q_hi)
        E_scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

        # Calibrated quantile
        n_cal = len(E_scores)
        level = (1 - self.alpha_target) * (1 + 1 / n_cal)
        self.cqr_Q = np.quantile(E_scores, min(level, 1.0))
        print(f"[CQR] Calibration quantile Q = {self.cqr_Q:.3f}")

        # ---- 4. Evaluate on test set ----
        y_pred_point = self.point_model.predict(X_test)
        q_lo_test = self.quantile_lo_model.predict(X_test) - self.cqr_Q
        q_hi_test = self.quantile_hi_model.predict(X_test) + self.cqr_Q

        # Clamp lower bound
        q_lo_test = np.maximum(q_lo_test, 0)

        mae = mean_absolute_error(y_test, y_pred_point)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_point))
        medae = median_absolute_error(y_test, y_pred_point)
        r2 = r2_score(y_test, y_pred_point)

        # Coverage
        coverage = np.mean((y_test >= q_lo_test) & (y_test <= q_hi_test))
        mpiw = np.mean(q_hi_test - q_lo_test)

        # SUR > 5min
        sur_5m = np.mean((y_test - y_pred_point) > 5) * 100

        # DAES
        daes = compute_daes(y_test, y_pred_point, gamma_threshold=10.0, alpha_asym=self.alpha_asym)

        # Winkler Score
        winkler = self._compute_winkler(y_test, q_lo_test, q_hi_test, self.alpha_target)

        print(f"\n[Stage-2 Wait-Time Predictor - Evaluation Results]")
        print(f"  MAE:       {mae:.2f} min")
        print(f"  RMSE:      {rmse:.2f} min")
        print(f"  MedAE:     {medae:.2f} min")
        print(f"  R²:        {r2:.4f}")
        print(f"  SUR>5m:    {sur_5m:.1f}%")
        print(f"  DAES:      {daes:.3f}")
        print(f"  PICP:      {coverage * 100:.1f}%")
        print(f"  MPIW:      {mpiw:.2f} min")
        print(f"  Winkler:   {winkler:.2f}")

        # ---- 5. Initialize SHAP explainer ----
        self.shap_explainer = shap.TreeExplainer(self.point_model)

        # ---- 6. Save models ----
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(self.point_model, os.path.join(save_dir, "stage2_point_model.pkl"))
            joblib.dump(self.quantile_lo_model, os.path.join(save_dir, "stage2_quantile_lo.pkl"))
            joblib.dump(self.quantile_hi_model, os.path.join(save_dir, "stage2_quantile_hi.pkl"))
            joblib.dump(self.cqr_Q, os.path.join(save_dir, "cqr_calibration_Q.pkl"))
            print(f"\n  Models saved to: {save_dir}")

        return {
            "mae": mae, "rmse": rmse, "medae": medae, "r2": r2,
            "sur_5m": sur_5m, "daes": daes,
            "picp": coverage * 100, "mpiw": mpiw, "winkler": winkler,
            "y_test": y_test, "y_pred": y_pred_point,
            "q_lo": q_lo_test, "q_hi": q_hi_test,
        }

    def predict(self, X) -> dict:
        """
        Generate wait-time prediction with CQR+ACI interval and SHAP explanation.
        """
        point = self.point_model.predict(X)
        q_lo = self.quantile_lo_model.predict(X) - self.cqr_Q
        q_hi = self.quantile_hi_model.predict(X) + self.cqr_Q
        q_lo = np.maximum(q_lo, 0)

        return {
            "point": point,
            "lower": q_lo,
            "upper": q_hi,
        }

    def predict_with_shap(self, X, feature_names: list = None) -> dict:
        """Predict with SHAP explanations."""
        result = self.predict(X)

        if self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(X)
            result["shap_values"] = shap_values
            result["shap_base"] = self.shap_explainer.expected_value

            if feature_names:
                result["feature_names"] = feature_names

        return result

    def aci_update(self, y_true: float, predicted_lower: float, predicted_upper: float):
        """
        Online ACI update step.
        Updates alpha_t based on whether the true value fell inside the interval.
        """
        err_t = 1.0 if (y_true < predicted_lower or y_true > predicted_upper) else 0.0
        self.aci_alpha_t = self.aci_alpha_t + self.aci_gamma * (self.alpha_target - err_t)
        self.aci_alpha_t = np.clip(self.aci_alpha_t, 0.001, 0.5)

        # Update CQR quantile based on new alpha
        # (In production, would recompute from calibration scores)
        return self.aci_alpha_t

    @staticmethod
    def _compute_winkler(y_true, lower, upper, alpha):
        """Compute Winkler Interval Score."""
        width = upper - lower
        penalty_lo = (2 / alpha) * np.maximum(lower - y_true, 0)
        penalty_hi = (2 / alpha) * np.maximum(y_true - upper, 0)
        return np.mean(width + penalty_lo + penalty_hi)

    @staticmethod
    def format_shap_explanation(shap_values_row, feature_names, top_k=4):
        """
        Format SHAP values into patient-friendly explanation strings.
        """
        if len(shap_values_row.shape) > 1:
            shap_vals = shap_values_row[0]
        else:
            shap_vals = shap_values_row

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_k]

        explanations = []
        FRIENDLY_NAMES = {
            "queue_length": "patients ahead in queue",
            "expected_queued_workload": "total workload ahead",
            "cond_remaining_active_time": "time until doctor finishes current patient",
            "physician_pact": "doctor's avg consultation speed",
            "physician_pnp": "doctor's hourly patient throughput",
            "fluid_estimator_1": "queue-to-capacity ratio",
            "rolling_wait_last_5": "recent patients' wait times",
            "queued_high_acuity_count": "urgent cases ahead",
            "visit_no": "visit type (new vs follow-up)",
            "patient_acuity_esi": "patient triage priority",
            "queue_new_ratio": "fraction of first-time visits in queue",
            "pred_own_duration": "predicted consultation length",
            "arrival_rate_1h": "current arrival rate",
            "time_zone_smoothness": "peak rush hour",
            "sin_time": "time of day",
            "cos_time": "time of day",
            "is_walk_in": "walk-in vs scheduled",
            "comorbidity_count": "patient health complexity",
            "busy_doctors": "doctors currently busy",
            "active_doctors": "available doctors",
        }

        for idx in sorted_idx:
            feat = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
            val = shap_vals[idx]
            friendly = FRIENDLY_NAMES.get(feat, feat.replace("_", " "))
            sign = "+" if val > 0 else ""
            explanations.append({
                "feature": feat,
                "friendly_name": friendly,
                "shap_value": float(val),
                "direction": "wait_increasing" if val > 0 else "wait_decreasing",
                "text": f"{sign}{val:.1f} min: {friendly}",
            })

        return explanations


# ==============================================================
# LSTM Sequence Encoder (Section 6.2)
# ==============================================================

class LSTMSequenceEncoder:
    """
    LSTM encoder for sequential delay-history over last K=10 queue events.
    Captures congestion waves across consecutive arrivals.
    """

    def __init__(self, input_size: int = 5, hidden_size: int = 32, seq_len: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.model = None

    def build_sequences(self, df: pd.DataFrame) -> tuple:
        """Build sequences of last K queue events for LSTM."""
        completed = df[df["status"] == "COMPLETED"].sort_values(
            ["session_id", "called_in_time"]
        ).copy()

        seq_features = [
            "actual_wait_time", "actual_service_duration",
            "queue_length", "esi_acuity", "doctor_id"
        ]

        sequences = []
        targets = []

        for session_id, session_df in completed.groupby("session_id"):
            values = session_df[seq_features].values.astype(np.float32)
            wait_times = session_df["actual_wait_time"].values.astype(np.float32)

            for i in range(self.seq_len, len(values)):
                seq = values[i - self.seq_len:i]
                target = wait_times[i]
                sequences.append(seq)
                targets.append(target)

        if len(sequences) == 0:
            return np.zeros((1, self.seq_len, len(seq_features))), np.zeros(1)

        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 30, save_path: str = None):
        """Train LSTM model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            print("[LSTM] PyTorch not available, skipping LSTM training.")
            return None

        X, y = self.build_sequences(df)
        print(f"[LSTM] Training on {len(X)} sequences, seq_len={self.seq_len}")

        # Normalize
        X_mean, X_std = X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-8
        X_norm = (X - X_mean) / X_std
        y_mean, y_std = y.mean(), y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        # Split
        n_train = int(0.8 * len(X_norm))
        X_train = torch.tensor(X_norm[:n_train])
        y_train = torch.tensor(y_norm[:n_train])
        X_val = torch.tensor(X_norm[n_train:])
        y_val = torch.tensor(y_norm[n_train:])

        # Model
        class WaitLSTM(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n[-1]).squeeze(-1)

        model = WaitLSTM(self.input_size, self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        best_val_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                # Denormalize for MAE
                val_pred_real = val_pred.numpy() * y_std + y_mean
                y_val_real = y_val.numpy() * y_std + y_mean
                val_mae = np.mean(np.abs(val_pred_real - y_val_real))
                print(f"  Epoch {epoch + 1}/{epochs} - Val MAE: {val_mae:.2f} min")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).numpy() * y_std + y_mean
            y_val_real = y_val.numpy() * y_std + y_mean
            mae = np.mean(np.abs(val_pred - y_val_real))
            rmse = np.sqrt(np.mean((val_pred - y_val_real) ** 2))

        print(f"[LSTM] Final - MAE: {mae:.2f} min, RMSE: {rmse:.2f} min")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "X_mean": X_mean, "X_std": X_std,
                "y_mean": y_mean, "y_std": y_std,
            }, save_path)
            print(f"  LSTM saved to: {save_path}")

        self.model = model
        return {"mae": mae, "rmse": rmse}


# ==============================================================
# Full Training Pipeline
# ==============================================================

def train_full_pipeline(data_path: str = None, models_dir: str = None):
    """
    Train the complete two-stage ML pipeline.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated_clinic_data.parquet")
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")

    print("=" * 60)
    print("CAREOS Module 1 - Training Two-Stage ML Pipeline")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records")

    # Compute features if not already present
    if "total_expected_workload_ahead" not in df.columns:
        print("\nComputing queue-mining features (16 groups, 28 variables)...")
        df = compute_queue_features(df)

    # Stage 1
    print("\n" + "=" * 60)
    print("STAGE 1: Consultation Duration Predictor")
    print("=" * 60)
    stage1 = Stage1DurationPredictor()
    stage1_results = stage1.train(df, save_path=os.path.join(models_dir, "stage1_duration_predictor.pkl"))

    # Stage 2
    print("\n" + "=" * 60)
    print("STAGE 2: Wait-Time Predictor (CQR + ACI + TreeSHAP)")
    print("=" * 60)
    stage2 = Stage2WaitTimePredictor(alpha_asym=0.8, alpha_target=0.10, aci_gamma=0.03)
    stage2_results = stage2.train(df, save_dir=models_dir)

    # LSTM
    print("\n" + "=" * 60)
    print("LSTM Sequence Encoder")
    print("=" * 60)
    lstm = LSTMSequenceEncoder()
    lstm_results = lstm.train(df, save_path=os.path.join(models_dir, "lstm_encoder.pt"))

    # Save evaluation results
    results_summary = {
        "stage1": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in stage1_results.items()},
        "stage2": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in stage2_results.items() if k not in ["y_test", "y_pred", "q_lo", "q_hi"]},
        "lstm": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in lstm_results.items()} if lstm_results else {},
    }
    joblib.dump(results_summary, os.path.join(models_dir, "evaluation_results.pkl"))

    import json
    metrics_json_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Metrics JSON saved to: {metrics_json_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return stage1, stage2, lstm, results_summary


if __name__ == "__main__":
    train_full_pipeline()
