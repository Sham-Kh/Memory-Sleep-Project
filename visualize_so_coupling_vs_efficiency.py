import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


SLEEP_STAGES = {1, 2, 3, 4, 5}


def load_hypnogram(hyp_path: Path) -> np.ndarray:
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
        raise ValueError(f"No numeric stages found in {hyp_path}")
    return np.asarray(values, dtype=int)


def running_sleep_efficiency(hypno: np.ndarray) -> np.ndarray:
    is_sleep = np.isin(hypno, list(SLEEP_STAGES)).astype(int)
    running_tst = np.cumsum(is_sleep)
    running_tib = np.arange(1, len(hypno) + 1)
    return running_tst / running_tib


def coupling_strength_from_phase(phases_deg: pd.Series) -> pd.Series:
    """
    Convert phase to a [0, 1] coupling-strength proxy.
    Higher means phase is closer to the recording's preferred SO-spindle phase.
    """
    phase_rad = np.deg2rad(phases_deg.astype(float))
    preferred_phase = np.angle(np.mean(np.exp(1j * phase_rad)))
    strength = (1.0 + np.cos(phase_rad - preferred_phase)) / 2.0
    return pd.Series(strength, index=phases_deg.index)


def main(features_csv: Path, hyp_path: Path, output_png: Path) -> None:
    df = pd.read_csv(features_csv)
    required_cols = {"epoch_index", "so_spindle_coupling_phase_deg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    hypno = load_hypnogram(hyp_path)
    se_running = running_sleep_efficiency(hypno)

    epoch_idx = df["epoch_index"].to_numpy(dtype=int)
    valid = (epoch_idx >= 0) & (epoch_idx < len(se_running))
    if not np.any(valid):
        raise ValueError("No valid epoch_index values aligned to hypnogram length.")

    df = df.loc[valid].copy()
    df["sleep_efficiency"] = se_running[epoch_idx[valid]]
    df = df.dropna(subset=["so_spindle_coupling_phase_deg"])
    if df.empty:
        raise ValueError("No non-null SO-spindle coupling phase values to visualize.")

    df["coupling_strength"] = coupling_strength_from_phase(
        df["so_spindle_coupling_phase_deg"]
    )

    threshold = float(df["sleep_efficiency"].median())
    df["efficiency_group"] = np.where(
        df["sleep_efficiency"] >= threshold, "High efficiency", "Low efficiency"
    )

    high = df.loc[df["efficiency_group"] == "High efficiency", "coupling_strength"]
    low = df.loc[df["efficiency_group"] == "Low efficiency", "coupling_strength"]
    if len(low) < 2 or len(high) < 2:
        raise ValueError(
            "Need at least 2 epochs in each group for a statistical test."
        )

    # Non-parametric group comparison between coupling strengths.
    u_stat, p_value = mannwhitneyu(low.to_numpy(), high.to_numpy(), alternative="two-sided")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].boxplot(
        [low.to_numpy(), high.to_numpy()],
        labels=["Low efficiency", "High efficiency"],
        patch_artist=True,
        boxprops={"facecolor": "#c6dbef"},
        medianprops={"color": "black"},
    )
    axes[0].set_ylabel("SO-Spindle Coupling Strength (0-1)")
    axes[0].set_title("Coupling Strength by Sleep Efficiency Group")
    axes[0].text(
        0.5,
        0.97,
        f"Mann-Whitney U p = {p_value:.3e}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    means = [float(low.mean()), float(high.mean())]
    sems = [
        float(low.std(ddof=1) / np.sqrt(len(low))) if len(low) > 1 else 0.0,
        float(high.std(ddof=1) / np.sqrt(len(high))) if len(high) > 1 else 0.0,
    ]
    axes[1].bar(
        ["Low efficiency", "High efficiency"],
        means,
        yerr=sems,
        capsize=5,
        color=["#9ecae1", "#3182bd"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean Coupling Strength")
    axes[1].set_title("Mean +/- SEM")

    fig.suptitle(
        f"SO-Spindle Coupling vs Sleep Efficiency (Median split = {threshold:.3f})"
    )
    fig.savefig(output_png, dpi=300)

    print(f"Saved plot to: {output_png}")
    print(f"Median sleep efficiency threshold: {threshold:.4f}")
    print(f"Low group n={len(low)}, mean coupling strength={means[0]:.4f}")
    print(f"High group n={len(high)}, mean coupling strength={means[1]:.4f}")
    print(f"Mann-Whitney U statistic: {u_stat:.4f}")
    print(f"Mann-Whitney U p-value: {p_value:.6g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SO-spindle coupling strength in high vs low sleep efficiency epochs."
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
        help="Path to hypnogram text file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("so_coupling_vs_sleep_efficiency.png"),
        help="Output figure path",
    )
    args = parser.parse_args()
    main(args.features, args.hyp, args.out)
