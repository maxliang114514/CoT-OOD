# Models

Place the base model weights here:

```
models/
├── llama2-7b/       # Llama2-7B (HuggingFace format)
└── qwen2.5-3b/      # Qwen2.5-3B (HuggingFace format)
```

Scripts default to `${ROOT_DIR}/models/llama2-7b` and `${ROOT_DIR}/models/qwen2.5-3b`; override with `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/your/model
```

> Model weights are large and git-ignored.
