# Operation Ablation (No-CoT Control)

## Paper Reference

- **Table 2b**: No-CoT control for the operation-ablation experiment.

Corresponds to `table1_main_cot/scripts/run_ablation.sh` (CoT side), but uses the **No-CoT** format (no intermediate steps, direct `Final Answer`). It shows that without CoT the model cannot generalize operations seen in single-step data to multi-step combinations.

## Layout

```
ablation_nocot/
├── README.md
├── prompts/
│   └── prompt_nocot_atom.txt # No-CoT prompt
├── results/                  # Results
└── scripts/
    ├── run_pipeline.sh       # Main script (ablation_no_*)
    ├── prepare_data.py       # Dataset format conversion (optional)
    └── predict.sh            # Single-model vLLM inference
```

## Usage

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh full
```

Runs the 5 `ablation_no_{reverse,substitute,duplicate,deduplicate,rotate}` experiments, seed 42.

> The single-step experiment belongs to the main experiment (see `t_scaling_nocot`); this directory only contains the No-CoT operation ablation.

Supports `train_only` / `test_only` / `full` (default).

## Output

Results go to `results/` (directory names carry a `_nocot` suffix).

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 2(b) - No-CoT, one operation excluded:

| Excluded | 3-step | 5-step | 7-step | 9-step |
|---|---|---|---|---|
| Reverse | 53.73+-0.50 | 30.33+-0.74 | 16.00+-1.18 | 5.53+-0.47 |
| Deduplicate | 40.40+-0.65 | 16.93+-0.66 | 6.80+-0.43 | 1.53+-0.09 |
| Substitute | 42.87+-0.81 | 21.40+-0.65 | 11.20+-0.33 | 5.07+-0.50 |
| Duplicate | 38.87+-1.23 | 17.20+-0.75 | 11.07+-1.09 | 3.40+-0.57 |
| Rotate | 55.47+-0.41 | 35.60+-1.84 | 22.67+-1.09 | 9.73+-0.93 |
