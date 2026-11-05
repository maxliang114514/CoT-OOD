# N / M Scaling (Sample Size / Context Length)

## Paper Reference

- **n-scaling (training sample size)**: Figure 3b, Table 6. Trains on `train_3step` subsampled at 10% / 20% / 50% / 100% and measures error rate vs. sample size.
- **m-scaling (context length / few-shot)**: Figure 3c, Table 7. Varies the number of in-context examples (0/2/3-shot) in the prompt and measures the effect on long-chain reasoning.

## Layout

```
nm_scaling/
├── README.md
├── datasets/                 # n-scaling subsets (n_scaling_{10,20,50,100}pct.jsonl)
├── prompts/                  # Prompt templates
│   ├── prompt_cot_atom.txt   #   Standard CoT prompt (1-shot, n-scaling)
│   └── prompt_nshot_{0,2,3}.txt  # few-shot prompts (m-scaling)
├── results/                  # Results
└── scripts/
    ├── run_nm_scaling.sh           # Main script (n_scaling / m_scaling)
    ├── generate_m_scaling_prompts.py   # Generate few-shot prompts
    └── generate_m_scaling_datasets.py  # Generate m-scaling training data
```

## Usage

```bash
cd scripts

# All (n-scaling + m-scaling)
CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh all

# n-scaling only (10% / 20% / 50%)
CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh n_scaling

# m-scaling only (0/2/3-shot)
CUDA_VISIBLE_DEVICES=0 bash run_nm_scaling.sh m_scaling
```

## Output

Results go to `results/` (e.g. `n_scaling_<frac>pct_<timestamp>`, `m_scaling_<n>shot_<timestamp>`).

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 6 - training-sample scaling (n), 3-step training data:

| Proportion | 3-step | 5-step | 7-step |
|---|---|---|---|
| 10% | 91.93+-1.32 | 78.00+-1.18 | 58.73+-2.81 |
| 20% | 95.33+-0.41 | 84.27+-1.27 | 65.87+-2.33 |
| 50% | 98.13+-0.38 | 91.40+-1.30 | 80.40+-3.56 |
| 100% | 99.33+-0.25 | 95.73+-0.68 | 88.87+-1.54 |

Table 7 - inference-time examples:

| Shots | 3-step | 5-step | 7-step |
|---|---|---|---|
| 0-shot | 99.27+-0.09 | 93.40+-2.83 | 82.53+-1.76 |
| 1-shot | 99.33+-0.25 | 95.73+-0.68 | 88.87+-1.54 |
| 2-shot | 99.33+-0.09 | 96.60+-0.16 | 90.93+-1.32 |
| 3-shot | 99.07+-0.19 | 95.73+-0.66 | 89.93+-0.81 |
