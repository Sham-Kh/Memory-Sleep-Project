## Clinical Lab Use: Automated Memory Consolidation Risk Screening

This pipeline can support a clinical sleep lab by turning EEG-derived N2/N3 features into a fast, reproducible risk screen for possible memory-consolidation vulnerability.

### Why this is clinically useful

- Spindle density and SO-spindle coupling are key physiological signatures linked to overnight memory processing.
- Manual spindle/SO scoring is time-consuming and inter-rater dependent; automated extraction reduces turnaround time and variability.
- The Random Forest model produces a transparent ranking of which neurophysiological signal is driving predicted sleep-efficiency quality in each dataset.

### Example workflow in a lab

1. Record overnight PSG and export EDF/REC plus hypnogram.
2. Run `extract_sleep_features.py` to compute per-epoch spindle and SO-coupling features in N2/N3.
3. Run `train_sleep_efficiency_rf.py` (or a pre-trained model) to estimate sleep-quality patterns and feature contribution.
4. Run `visualize_so_coupling_vs_efficiency.py` to compare coupling strength between low- and high-efficiency epochs.
5. Flag patients with low spindle density and/or weak SO-spindle coupling alignment as higher "Memory Consolidation Risk" for clinical follow-up.

### How to interpret outputs for screening

- **Primary risk indicators**
  - Persistently low `spindle_density_per_min`
  - Reduced SO-spindle coupling strength in lower-efficiency epochs
- **Model signal attribution**
  - Feature importance identifies whether spindle or slow-wave-related markers better explain reduced sleep quality in that patient.
- **Suggested action**
  - Use this as a triage layer for prioritizing full cognitive/sleep specialist review, not as a standalone diagnosis.

### Practical safeguards

- Validate thresholds and model behavior on local historical data before deployment.
- Recalibrate for different age groups, medications, comorbidities, and PSG hardware.
- Retain clinician oversight and include quality-control checks for signal artifacts and scoring errors.