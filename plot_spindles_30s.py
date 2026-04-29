import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa


EPOCH_SEC = 30
N2_STAGE = 2
N3_STAGE = 3


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
        raise ValueError(f"No numeric stage codes found in {hyp_path}")
    return np.asarray(values, dtype=int)


def pick_n2n3_window(hyp_epoch: np.ndarray) -> int:
    """Return first epoch index in N2/N3, fallback to 0."""
    idx = np.where(np.isin(hyp_epoch, [N2_STAGE, N3_STAGE]))[0]
    return int(idx[0]) if len(idx) > 0 else 0


def main(rec_path: Path, hyp_path: Path, out_png: Path, epoch_index: int | None) -> None:
    raw = mne.io.read_raw_edf(rec_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])

    eeg_chs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_chs) == 0:
        raise RuntimeError("No EEG channels found.")
    ch_name = raw.ch_names[eeg_chs[0]]

    data = raw.get_data(picks=[ch_name]).squeeze()
    times = raw.times

    hyp_epoch = load_hypnogram(hyp_path)
    hyp_sample = yasa.hypno_upsample_to_data(
        hypno=hyp_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
    )

    hyp_for_detect = hyp_sample.copy()
    hyp_for_detect[~np.isin(hyp_for_detect, [N2_STAGE, N3_STAGE])] = -99

    sp = yasa.spindles_detect(
        data=data,
        sf=sf,
        ch_names=[ch_name],
        hypno=hyp_for_detect,
        include=(N2_STAGE, N3_STAGE),
        verbose=False,
    )
    sp_df = sp.summary() if sp is not None else pd.DataFrame()

    if epoch_index is None:
        epoch_index = pick_n2n3_window(hyp_epoch)

    start_sec = float(epoch_index * EPOCH_SEC)
    end_sec = float(start_sec + EPOCH_SEC)

    in_window = (times >= start_sec) & (times < end_sec)
    if not np.any(in_window):
        raise ValueError("Selected 30-second window is outside EEG duration.")

    x = times[in_window] - start_sec
    y = data[in_window]

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(x, y, color="black", linewidth=1.0, label=f"EEG ({ch_name})")

    n_highlight = 0
    if not sp_df.empty and {"Start", "End"}.issubset(sp_df.columns):
        sp_win = sp_df[(sp_df["End"] > start_sec) & (sp_df["Start"] < end_sec)].copy()
        for _, row in sp_win.iterrows():
            s = max(float(row["Start"]), start_sec) - start_sec
            e = min(float(row["End"]), end_sec) - start_sec
            if e > s:
                ax.axvspan(s, e, color="#e41a1c", alpha=0.25)
                n_highlight += 1

    ax.set_xlim(0, EPOCH_SEC)
    ax.set_xlabel("Time in 30-second window (s)")
    ax.set_ylabel("Amplitude (uV, raw units from EDF)")
    ax.set_title(
        f"Detected spindles in 30s window (epoch {epoch_index}, stage {int(hyp_epoch[epoch_index]) if epoch_index < len(hyp_epoch) else 'NA'})"
    )
    ax.grid(alpha=0.2)

    legend_text = f"Highlighted spindles: {n_highlight}"
    ax.legend([legend_text], loc="upper right", frameon=True)

    fig.savefig(out_png, dpi=300)
    print(f"EEG channel used: {ch_name}")
    print(f"Window: {start_sec:.1f}s to {end_sec:.1f}s")
    print(f"Spindles highlighted in window: {n_highlight}")
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 30 seconds of EEG and highlight detected N2/N3 spindles."
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
        default=Path("spindles_30s_plot.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Optional epoch index (30s epochs). If omitted, first N2/N3 epoch is used.",
    )
    args = parser.parse_args()
    main(args.rec, args.hyp, args.out, args.epoch)
