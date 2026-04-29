import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yasa


EPOCH_SEC = 30
N2_STAGE = 2
N3_STAGE = 3


def load_hypnogram(hyp_path: Path) -> np.ndarray:
    """Load hypnogram with one stage code per epoch (text format)."""
    with hyp_path.open("r", encoding="utf-8", errors="ignore") as f:
        values = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(float(line)))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"Could not parse any numeric stage codes from {hyp_path}")
    return np.asarray(values, dtype=int)


def circular_mean_deg(phases_deg: pd.Series) -> float:
    if phases_deg.empty:
        return np.nan
    radians = np.deg2rad(phases_deg.to_numpy(dtype=float))
    mean_angle = np.angle(np.mean(np.exp(1j * radians)))
    return float(np.rad2deg(mean_angle))


def main(rec_path: Path, hyp_path: Path, output_csv: Path) -> None:
    raw = mne.io.read_raw_edf(rec_path, preload=True, verbose=False)
    sf = float(raw.info["sfreq"])

    eeg_chs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_chs) == 0:
        raise RuntimeError("No EEG channels found in recording.")
    ch_name = raw.ch_names[eeg_chs[0]]

    data = raw.get_data(picks=[ch_name]).squeeze()
    hypno_epoch = load_hypnogram(hyp_path)
    hypno_sample = yasa.hypno_upsample_to_data(
        hypno=hypno_epoch, sf_hypno=1 / EPOCH_SEC, data=data, sf_data=sf
    )

    n2n3_mask = np.isin(hypno_sample, [N2_STAGE, N3_STAGE])

    # Run detections only on N2/N3 samples by masking all other stages.
    # Stages outside N2/N3 get a sentinel value not included in the detector's include tuple.
    hypno_for_detect = hypno_sample.copy()
    hypno_for_detect[~n2n3_mask] = -99

    sp = yasa.spindles_detect(
        data=data,
        sf=sf,
        ch_names=[ch_name],
        hypno=hypno_for_detect,
        include=(N2_STAGE, N3_STAGE),
        verbose=False,
    )
    sw = yasa.sw_detect(
        data=data,
        sf=sf,
        ch_names=[ch_name],
        hypno=hypno_for_detect,
        include=(N2_STAGE, N3_STAGE),
        coupling=True,
        verbose=False,
    )

    sp_df = sp.summary() if sp is not None else pd.DataFrame()
    sw_df = sw.summary() if sw is not None else pd.DataFrame()

    n_samples = data.shape[0]
    n_epochs = int(np.floor(n_samples / (EPOCH_SEC * sf)))
    epoch_stages = hypno_epoch[:n_epochs]

    rows = []
    for ep in range(n_epochs):
        stage = int(epoch_stages[ep]) if ep < len(epoch_stages) else -1
        if stage not in (N2_STAGE, N3_STAGE):
            continue

        start_sec = ep * EPOCH_SEC
        end_sec = (ep + 1) * EPOCH_SEC

        if not sp_df.empty and "Start" in sp_df.columns:
            sp_ep = sp_df[(sp_df["Start"] >= start_sec) & (sp_df["Start"] < end_sec)]
            spindle_count = int(len(sp_ep))
        else:
            spindle_count = 0
        spindle_density = spindle_count / (EPOCH_SEC / 60.0)

        coupling_phase = np.nan
        if not sw_df.empty and "Start" in sw_df.columns and "PhaseAtSigmaPeak" in sw_df.columns:
            sw_ep = sw_df[(sw_df["Start"] >= start_sec) & (sw_df["Start"] < end_sec)]
            coupling_phase = circular_mean_deg(sw_ep["PhaseAtSigmaPeak"].dropna())

        rows.append(
            {
                "epoch_index": ep,
                "stage": stage,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "spindle_count": spindle_count,
                "spindle_density_per_min": spindle_density,
                "so_spindle_coupling_phase_deg": coupling_phase,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"Channel used: {ch_name}")
    print(f"Wrote {len(out_df)} N2/N3 epochs to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract N2/N3 spindle and slow-wave coupling features per epoch."
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
        help="Path to hypnogram text file (one stage code per epoch)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sleep_features.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args.rec, args.hyp, args.out)
