import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa


EPOCH_SEC = 30
N3_STAGE = 3


def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """Load text hypnogram with one numeric stage code per epoch."""
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
        raise ValueError(f"Could not parse numeric stage codes from {hyp_path}")
    return np.asarray(values, dtype=int)


def main(rec_path: Path, hyp_path: Path, out_png: Path) -> None:
    raw = mne.io.read_raw_edf(rec_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) == 0:
        raise RuntimeError("No EEG channels found in the recording.")
    ch_name = raw.ch_names[eeg_picks[0]]

    data = raw.get_data(picks=[ch_name]).squeeze()
    hyp_epoch = load_hypnogram(hyp_path)
    hyp_sample = yasa.hypno_upsample_to_data(
        hypno=hyp_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
    )

    # Keep only N3 for coupling analysis.
    hyp_for_detect = hyp_sample.copy()
    hyp_for_detect[hyp_for_detect != N3_STAGE] = -99

    sw = yasa.sw_detect(
        data=data,
        sf=sf,
        ch_names=[ch_name],
        hypno=hyp_for_detect,
        include=(N3_STAGE,),
        coupling=True,
        verbose=False,
    )
    if sw is None:
        raise RuntimeError("No slow waves detected in N3. Try a different channel or file.")

    sw_df = sw.summary()
    required_cols = {"PhaseAtSigmaPeak", "SigmaPeak"}
    if sw_df.empty or not required_cols.issubset(sw_df.columns):
        raise RuntimeError(
            "Coupling metrics not found. Ensure YASA coupling columns "
            "'PhaseAtSigmaPeak' and 'SigmaPeak' are available."
        )

    phase = sw_df["PhaseAtSigmaPeak"].dropna().to_numpy()
    amp = sw_df.loc[sw_df["PhaseAtSigmaPeak"].notna(), "SigmaPeak"].to_numpy()
    if len(phase) < 5:
        raise RuntimeError("Not enough N3 coupling events to plot (need at least 5).")

    # YASA 0.7+ removed yasa.plot_coupling; use polar scatter (phase vs σ amplitude).
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    ax.scatter(np.deg2rad(phase), amp, alpha=0.55, s=22, c=amp, cmap="viridis")
    mean_rad = np.angle(np.mean(np.exp(1j * np.deg2rad(phase))))
    ax.plot([mean_rad, mean_rad], [0, float(np.max(amp))], color="crimson", lw=2)
    ax.set_title(
        f"N3 SO-Spindle Coupling ({ch_name})\nPhase where spindle amplitude peaks",
        fontsize=11,
        pad=16,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)

    print(f"Channel used: {ch_name}")
    print(f"N3 slow waves with coupling: {len(phase)}")
    print(f"Saved coupling plot to: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Polar plot of N3 slow-oscillation phase vs spindle (σ) peak amplitude."
        )
    )
    parser.add_argument(
        "--rec",
        type=Path,
        default=Path(r"C:\Users\shamm\Downloads\sc4002e0.rec"),
        help="Path to EDF/REC file",
    )
    parser.add_argument(
        "--hyp",
        type=Path,
        default=Path(r"C:\Users\shamm\Downloads\sc4002e0.hyp"),
        help="Path to hypnogram file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("n3_so_spindle_coupling.png"),
        help="Output coupling plot path",
    )
    args = parser.parse_args()
    main(args.rec, args.hyp, args.out)
