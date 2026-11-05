# Datasets

Shared base datasets used by all experiments. All scripts reference this directory through repository-relative paths.

## Files

| File | Description | Size |
|------|-------------|------|
| `train_1step.jsonl` | Single-step training data | 5,000 |
| `train_3step.jsonl` | Three-step training data (formerly `multi_step_type7_v2`) | 1,667 |
| `train_mixed_3_5step.jsonl` | Mixed 3+5 step training (formerly `multi_step_type8_v2`) | 1,333 |
| `ablation_no_reverse.jsonl` | Ablation: Reverse excluded | 5,000 |
| `ablation_no_substitute.jsonl` | Ablation: Substitute excluded | 5,000 |
| `ablation_no_duplicate.jsonl` | Ablation: Duplicate excluded | 5,000 |
| `ablation_no_deduplicate.jsonl` | Ablation: Deduplicate excluded | 5,000 |
| `ablation_no_rotate.jsonl` | Ablation: Rotate excluded | 5,000 |
| `test_3step.jsonl` | 3-step test set (formerly `test_composition_3step`) | 500 |
| `test_5step.jsonl` | 5-step test set | 500 |
| `test_7step.jsonl` | 7-step test set | 500 |
| `test_9step.jsonl` | 9-step test set | 500 |

> Naming: `type7` / `type8` were early internal names for the **Three-Step** and **Mixed 3+5** training sets, renamed to `train_3step` / `train_mixed_3_5step`.

## Reference Scripts

The original data scripts shipped with the data package are kept here for reference (they contain server paths and are not used at runtime):

| File | Description |
|------|-------------|
| `data_generator.py` | Original data-generation core module (also at `src/data_generator.py`) |
| `generate_dataset.py` | Original base-dataset generation script |
| `generate_test_doubled.py` | Original test-set expansion script |
| `restore_test_sets.py` | Original test-set restoration script |

The actively used, path-adapted scripts live in `src/` and `experiments/*/scripts/`.

## Record Format

Each line is a JSON object:

```json
{
  "input": "δωωωδγ",
  "output": "γδδωωωωωωδδγ",
  "task_chain": "t1 → t3 → t5",
  "task_chain_readable": "Reverse → Duplicate → Rotate",
  "steps": 3,
  "reasoning_steps": [
    {"step": 1, "task": "Reverse", "input": "δωωωδγ", "output": "γδωωωδ"},
    {"step": 2, "task": "Duplicate", "input": "γδωωωδ", "output": "γγδδωωωωωωδδ"},
    {"step": 3, "task": "Rotate", "input": "γγδδωωωωωωδδ", "output": "γδδωωωωωωδδγ"}
  ]
}
```

- `input`: symbol sequence over {α, β, γ, δ, ε, ω}.
- `output`: final output sequence.
- `task_chain` / `task_chain_readable`: task chain (IDs / names).
- `reasoning_steps`: step-by-step reasoning used to build CoT training/test formats.
