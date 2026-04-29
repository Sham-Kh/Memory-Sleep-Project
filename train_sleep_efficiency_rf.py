import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


SLEEP_STAGES = {1, 2, 3, 4, 5}  # N1, N2, N3/N4, REM (common numeric coding)


def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """Load hypnogram text file containing one numeric stage code per epoch."""
    values = []
    with hyp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(float(line)))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric sleep stages found in {hyp_path}")
    return np.asarray(values, dtype=int)


def compute_sleep_efficiency(hypno: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Return:
      - Final nightly Sleep Efficiency = TST / TIB
      - Running sleep efficiency per epoch for supervised learning target
    """
    is_sleep = np.isin(hypno, list(SLEEP_STAGES)).astype(int)
    tib_epochs = len(hypno)
    tst_epochs = int(is_sleep.sum())
    nightly_efficiency = tst_epochs / tib_epochs if tib_epochs else np.nan

    running_tst = np.cumsum(is_sleep)
    running_tib = np.arange(1, tib_epochs + 1)
    running_efficiency = running_tst / running_tib
    return nightly_efficiency, running_efficiency


def main(features_csv: Path, hyp_path: Path, random_state: int) -> None:
    df = pd.read_csv(features_csv)
    hypno = load_hypnogram(hyp_path)
    nightly_efficiency, running_efficiency = compute_sleep_efficiency(hypno)

    if "epoch_index" not in df.columns:
        raise ValueError(
            "sleep_features.csv must include 'epoch_index' column so epochs can be "
            "matched to hypnogram-derived sleep efficiency."
        )

    # Map each epoch in sleep_features to a target value from the hypnogram.
    # We use running SE so each epoch has a learnable, non-constant target.
    epoch_idx = df["epoch_index"].to_numpy(dtype=int)
    valid_mask = (epoch_idx >= 0) & (epoch_idx < len(running_efficiency))
    if not np.any(valid_mask):
        raise ValueError("No valid epoch_index values match the hypnogram length.")

    df = df.loc[valid_mask].copy()
    df["sleep_efficiency_target"] = running_efficiency[epoch_idx[valid_mask]]

    feature_cols = [
        "spindle_density_per_min",
        "so_spindle_coupling_phase_deg",
        "stage",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[feature_cols]
    y = df["sleep_efficiency_target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=400, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    rf = model.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    spindle_signal_importance = float(importances.get("spindle_density_per_min", 0.0))
    slow_wave_signal_importance = float(
        importances.get("so_spindle_coupling_phase_deg", 0.0)
    )

    print("=== Sleep Efficiency ===")
    print(f"Nightly Sleep Efficiency (TST / TIB): {nightly_efficiency:.4f}")
    print()
    print("=== Random Forest Performance ===")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")
    print()
    print("=== Feature Importance ===")
    for feat, imp in importances.items():
        print(f"{feat}: {imp:.4f}")
    print()
    print("=== Neurophysiological Signal Comparison ===")
    if spindle_signal_importance > slow_wave_signal_importance:
        best = "Spindles (spindle_density_per_min)"
    elif slow_wave_signal_importance > spindle_signal_importance:
        best = "Slow-wave coupling (so_spindle_coupling_phase_deg)"
    else:
        best = "Tie"
    print(f"Spindles importance: {spindle_signal_importance:.4f}")
    print(f"Slow-wave importance: {slow_wave_signal_importance:.4f}")
    print(f"Best predictor of sleep quality: {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest model to predict sleep efficiency."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("sleep_features.csv"),
        help="Path to sleep_features.csv",
    )
    parser.add_argument(
        "--hyp",
        type=Path,
        default=Path(r"C:\Users\shamm\Downloads\sc4002e0.hyp"),
        help="Path to hypnogram file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and model",
    )
    args = parser.parse_args()
    main(args.features, args.hyp, args.seed)
