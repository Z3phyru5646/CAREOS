"""
CAREOS Module 1 - Complete Pipeline Runner
Runs everything: Simulation → Feature Engineering → Model Training → Evaluation
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    start_time = time.time()

    print("=" * 70)
    print("  CAREOS MODULE 1: Dynamic Outpatient Queue & Wait-Time Predictor")
    print("  Complete Pipeline Execution")
    print("=" * 70)

    # ============ Phase 1: Simulation ============
    print("\n" + "=" * 70)
    print("  PHASE 1: Running Calibrated SimPy Digital Twin Simulation")
    print("=" * 70)

    from src.simulation.clinic_simpy_twin import run_simulation_and_save

    data_path = os.path.join(PROJECT_ROOT, "data", "simulated_clinic_data.parquet")
    df = run_simulation_and_save(output_path=data_path, num_sessions=100, seed=42)

    # ============ Phase 2: Feature Engineering ============
    print("\n" + "=" * 70)
    print("  PHASE 2: Computing Queue-Mining Features (16 Groups, 28 Variables)")
    print("=" * 70)

    from src.features.queue_feature_engine import compute_queue_features

    df_featured = compute_queue_features(df)
    featured_path = os.path.join(PROJECT_ROOT, "data", "featured_clinic_data.parquet")
    df_featured.to_parquet(featured_path, index=False)
    sample_csv_path = os.path.join(PROJECT_ROOT, "data", "sample_clinic_data.csv")
    df_featured.head(200).to_csv(sample_csv_path, index=False)
    print(f"Featured dataset saved: {featured_path}")
    print(f"Sample CSV saved: {sample_csv_path}")
    print(f"Shape: {df_featured.shape}")

    # ============ Phase 3: Model Training ============
    print("\n" + "=" * 70)
    print("  PHASE 3: Training Two-Stage ML Pipeline")
    print("=" * 70)

    from src.models.train_pipeline import train_full_pipeline

    models_dir = os.path.join(PROJECT_ROOT, "models")
    stage1, stage2, lstm, results = train_full_pipeline(
        data_path=featured_path,
        models_dir=models_dir,
    )

    # ============ Summary ============
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n  Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Simulated patients:  {len(df)}")
    print(f"  Feature columns:     {len(df_featured.columns)}")

    if results.get("stage1"):
        s1 = results["stage1"]
        print(f"\n  Stage-1 Duration Predictor:")
        print(f"    MAE:  {s1.get('mae', 0):.2f} min")
        print(f"    RMSE: {s1.get('rmse', 0):.2f} min")
        print(f"    R²:   {s1.get('r2', 0):.4f}")

    if results.get("stage2"):
        s2 = results["stage2"]
        print(f"\n  Stage-2 Wait-Time Predictor:")
        print(f"    MAE:      {s2.get('mae', 0):.2f} min")
        print(f"    RMSE:     {s2.get('rmse', 0):.2f} min")
        print(f"    PICP:     {s2.get('picp', 0):.1f}%")
        print(f"    MPIW:     {s2.get('mpiw', 0):.2f} min")
        print(f"    Winkler:  {s2.get('winkler', 0):.2f}")
        print(f"    SUR>5m:   {s2.get('sur_5m', 0):.1f}%")
        print(f"    DAES:     {s2.get('daes', 0):.3f}")

    if results.get("lstm"):
        lstm_r = results["lstm"]
        print(f"\n  LSTM Sequence Encoder:")
        print(f"    MAE:  {lstm_r.get('mae', 0):.2f} min")
        print(f"    RMSE: {lstm_r.get('rmse', 0):.2f} min")

    print(f"\n  Saved artifacts:")
    print(f"    Data:    {data_path}")
    print(f"    Features: {featured_path}")
    print(f"    Models:  {models_dir}/")

    print("\n  To run the dashboard:")
    print(f"    streamlit run src/dashboard/streamlit_app.py")
    print("\n  To run the API:")
    print(f"    uvicorn src.api.main:app --reload")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
