import argparse
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa


EPOCH_SEC = 30
N2_STAGE = 2
N3_STAGE = 3


def load_hypnogram(hyp_path: Path) -> np.ndarray:
    stages = []
    with hyp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                stages.append(int(float(line)))
            except ValueError:
                continue
    if stages:
        return np.asarray(stages, dtype=int)

    # Fallback: many .hyp files are EDF-like hypnogram files.
    temp_dir = tempfile.TemporaryDirectory()
    hyp_edf = Path(temp_dir.name) / f"{hyp_path.stem}.edf"
    shutil.copyfile(hyp_path, hyp_edf)
    raw_hyp = mne.io.read_raw_edf(hyp_edf, preload=True, verbose=False)
    hyp_data = raw_hyp.get_data(picks=[0]).squeeze()
    temp_dir.cleanup()
    if hyp_data.size == 0:
        raise ValueError(f"No hypnogram values found in {hyp_path}")
    return np.rint(hyp_data).astype(int)


def hyp_to_annotations(hyp: np.ndarray) -> mne.Annotations:
    onset = np.arange(len(hyp), dtype=float) * EPOCH_SEC
    duration = np.full(len(hyp), EPOCH_SEC, dtype=float)
    desc = [f"Stage {int(s)}" for s in hyp]
    return mne.Annotations(onset=onset, duration=duration, description=desc)


def find_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    exact = [ch for ch in raw.ch_names if ch == requested]
    if exact:
        return exact[0]
    lowered = requested.lower()
    partial = [ch for ch in raw.ch_names if lowered in ch.lower()]
    if partial:
        return partial[0]
    raise ValueError(f"Requested channel '{requested}' not found in recording.")


def main(rec_path: Path, hyp_path: Path, out_png: Path, channel: str) -> None:
    rec_for_read = rec_path
    temp_dir = None
    if rec_path.suffix.lower() == ".rec":
        temp_dir = tempfile.TemporaryDirectory()
        rec_for_read = Path(temp_dir.name) / f"{rec_path.stem}.edf"
        shutil.copyfile(rec_path, rec_for_read)

    raw = mne.io.read_raw_edf(rec_for_read, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])

    hyp_epoch = load_hypnogram(hyp_path)
    raw.set_annotations(hyp_to_annotations(hyp_epoch))

    ch_name = find_channel(raw, channel)
    # MNE returns EEG in Volts; YASA expects microvolts.
    data = raw.get_data(picks=[ch_name]).squeeze() * 1e6
    times = raw.times

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
    sw = yasa.sw_detect(
        data=data,
        sf=sf,
        ch_names=[ch_name],
        hypno=hyp_for_detect,
        include=(N2_STAGE, N3_STAGE),
        coupling=False,
        verbose=False,
    )

    sp_df = sp.summary() if sp is not None else pd.DataFrame()
    sw_df = sw.summary() if sw is not None else pd.DataFrame()

    n2_epochs = np.where(hyp_epoch == N2_STAGE)[0]
    if len(n2_epochs) == 0:
        raise RuntimeError("No N2 epochs found in hypnogram.")
    ep = int(n2_epochs[0])
    if not sp_df.empty and {"Start", "End"}.issubset(sp_df.columns):
        for cand in n2_epochs:
            cand_start = float(cand * EPOCH_SEC)
            cand_end = cand_start + EPOCH_SEC
            has_sp = ((sp_df["End"] > cand_start) & (sp_df["Start"] < cand_end)).any()
            if has_sp:
                ep = int(cand)
                break
    start_sec = ep * EPOCH_SEC
    end_sec = start_sec + EPOCH_SEC

    mask = (times >= start_sec) & (times < end_sec)
    x = times[mask] - start_sec
    y = data[mask]

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(x, y, color="black", lw=1.0, label=f"EEG ({ch_name})")

    n_win_sp = 0
    if not sp_df.empty and {"Start", "End"}.issubset(sp_df.columns):
        sp_win = sp_df[(sp_df["End"] > start_sec) & (sp_df["Start"] < end_sec)]
        for _, row in sp_win.iterrows():
            s = max(float(row["Start"]), start_sec) - start_sec
            e = min(float(row["End"]), end_sec) - start_sec
            if e > s:
                ax.axvspan(s, e, color="#d62728", alpha=0.28)
                n_win_sp += 1

    ax.set_xlim(0, EPOCH_SEC)
    ax.set_xlabel("Time (s) within 30-second N2 window")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title("N2 EEG (30s) with detected spindle intervals highlighted")
    ax.grid(alpha=0.2)
    ax.text(
        0.01,
        0.96,
        f"Spindles in window: {n_win_sp}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.savefig(out_png, dpi=300)

    print(f"Channel used: {ch_name}")
    print(f"N2 window: {start_sec:.1f}s to {end_sec:.1f}s")
    print(f"Spindles detected total (N2+N3): {len(sp_df)}")
    print(f"Slow waves detected total (N2+N3): {len(sw_df)}")
    print(f"Spindles highlighted in plotted 30s N2 window: {n_win_sp}")
    print(f"Saved figure: {out_png}")
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview N2 30s EEG with spindle highlights and spindle/SW counts."
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
        default=Path("n2_spindles_preview.png"),
        help="Output plot path",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="Fpz-Cz",
        help="EEG channel name to analyze (default: Fpz-Cz)",
    )
    args = parser.parse_args()
    main(args.rec, args.hyp, args.out, args.channel)
