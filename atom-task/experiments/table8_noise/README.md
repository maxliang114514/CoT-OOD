# Noise Robustness

## Paper Reference

- **Table 8**: Robustness of CoT training to reasoning-step noise.

## Design

Noise is injected into reasoning steps and each dataset is trained independently:

| Dataset | Description |
|---------|-------------|
| `train_3step` | 3-step reasoning |
| `train_mixed_3_5step` | Mixed 3+5 step reasoning |
| `ablation_no_substitute` | Ablation without the Substitute operation |

Noise ratio p ∈ {0.05, 0.10, 0.20} (p=0 is the clean baseline).

**Injection strategy**: each reasoning step contains four elements (`Input_Sequence` / `Output_Sequence` / current `Task_Chain` / updated `Task_Chain`). For each selected step, two elements are modified by **Tampering** (replace with random symbols) or **Deletion** (set to `***`).

## Layout

```
noise_robust/
├── README.md
├── datasets/                 # Noise datasets (train_3step_noise_{0,5,10,20}.jsonl, etc.)
├── results/                  # Results
└── scripts/
    ├── run_pipeline.sh       # Main script
    ├── generate_datasets.py  # Generate noise datasets
    └── add_noise_to_cot.py   # Noise injection module
```

## Usage

```bash
cd scripts

# All 12 experiments (generates datasets first)
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh all

# train_3step series only
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh all_type7

# Single experiment
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh type7_noise_10 full
```

Supports `train_only` / `test_only` / `full` (default).

## Expected Outcome

Low noise (p=0.05) is absorbed without loss; at p=0.20 long-chain accuracy drops slightly but remains strong — the learned compositional patterns are resilient to localized inconsistencies.

## Output

Results go to `results/`.

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 8 - training-noise robustness (3-step / 5-step / 7-step):

| Noise ratio | Pure 3-step | Mix3+5 | w/o Substitute |
|---|---|---|---|
| p = 0.05 | 99.73+-0.09 / 97.80+-0.86 / 90.47+-2.46 | 99.33+-0.09 / 98.53+-0.34 / 94.87+-0.81 | 96.20+-0.65 / 92.33+-1.06 / 85.47+-2.80 |
| p = 0.10 | 99.47+-0.09 / 97.20+-0.59 / 91.67+-0.25 | 99.27+-0.34 / 98.87+-0.25 / 94.80+-0.49 | 95.80+-0.33 / 92.33+-1.27 / 86.73+-1.04 |
| p = 0.20 | 99.40+-0.00 / 96.60+-0.75 / 90.73+-1.89 | 99.13+-0.09 / 98.93+-0.62 / 94.80+-0.28 | 96.93+-0.77 / 92.47+-0.50 / 87.13+-1.97 |
