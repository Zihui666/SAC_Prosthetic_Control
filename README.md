# Assignment Project: SAC Prosthetic Control (Gymnasium)

> **Disclaimer**  
> This repository is primarily organized for a course/assignment submission. The implementation is **not production-quality**:
> - the codebase is intentionally lightweight and not fully refactored,
> - hyperparameters are not exhaustively tuned,
> - training is kept **small-scale** to finish within typical assignment time/compute limits.

This project runs a simple 3-phase reinforcement learning experiment (**SAC**) on Gymnasium `Reacher-v4`:

1. **Phase 1 — Normal training (baseline)**
2. **Phase 2 — “Motor damage” evaluation (no training)** via an action wrapper that injects *bias/weakness*
3. **Phase 3 — Adaptation / rehabilitation training** under the same motor damage, then re-evaluation

Key scripts:
- `SAC_Prosthetic_Control.py` — main pipeline (train/eval, export GIFs, plot learning curves)
- `plotting.py` — generates a clean zoomed plot from SB3 Monitor logs

---

## Layout (logs in a separate folder; outputs at repo root)

After running, your repository will look roughly like:

```
.
├── SAC_Prosthetic_Control.py
├── plotting.py
├── analysis.py
├── requirements.txt
├── README.md
├── logs/
│   ├── monitor_normal.csv.monitor.csv
│   └── monitor_adapt.csv.monitor.csv
├── Motor_Damage_Result.png
├── Calibration_Zoomed_Clean.png
├── 1_Normal_Control.gif
├── 2_Motor_Damage_Effect.gif
├── 3_Recovered_Control.gif
├── brain_phase1.zip
├── vec_stats_phase1.pkl
├── brain_phase3_adapted.zip
└── vec_stats_phase3.pkl
```

---

## Installation

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

### 1) Run the full 3-phase experiment
```bash
python SAC_Prosthetic_Control.py
```

This script will:
- write training logs under `logs/` (Monitor csv)
- export learning-curve figure(s) to the repo root (e.g., `Motor_Damage_Result.png`)
- export 3 GIFs to the repo root (baseline / damaged / recovered)
- save trained agents (`.zip`) and normalization stats (`.pkl`) to the repo root

### 2) Generate the clean zoomed plot (optional)
```bash
python plotting.py
```

### 3) Quantitative Analysis & Ablation Study
```bash
python analysis.py
```
To rigorously evaluate the impact of motor damage and the effectiveness of neuroplasticity (Phase 3 adaptation), we included a dedicated analysis script `analysis.py`. This script loads the pre-trained models (Phase 1 & Phase 3) and performs two key experiments:

### A. Performance Recovery (Bar Chart)
We compare the agent's performance across three distinct conditions (averaged over 30 episodes):

### B. Sensitivity Analysis (Ablation Study / Heatmap)
To understand which fault factor (Bias vs. Weakness) is more critical, we conducted a grid-search sensitivity analysis on the Phase 1 model.


---

## Quick knobs you may adjust

Inside `SAC_Prosthetic_Control.py`, you can quickly adjust:
- `TRAIN_STEPS` — number of training steps (kept small for the assignment)
- `DAMAGE_CONFIG = {"weakness": ..., "bias": ...}` — damage strength (strongly affects behavior)

---

## Notes
- The “motor damage” is implemented in `MotorDamageWrapper` by modifying one action dimension:
  `action[1] = action[1] * weakness + bias`.
- `VecNormalize` statistics are reused across phases for consistent observation normalization.
