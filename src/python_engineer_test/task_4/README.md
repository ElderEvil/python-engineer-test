# Task 4: VIPE Gaussian Splatting

Run VIPE Gaussian Splatting pipeline on image dataset.

## Usage

```bash
# Via CLI
pet task4 /path/to/dataset --workspace workspace --output demo.mp4

# Smoke-friendly (no GPU required): validate inputs and print commands
pet task4 /path/to/dataset --dry-run

# Direct module
python -m python_engineer_test.task_4.vipe_runner /path/to/dataset
```

## Options

| Option | Description |
|--------|-------------|
| `--workspace` | Working directory for VIPE (default: workspace) |
| `--output` | Output demo video (default: demo.mp4) |
| `--dry-run` | Validate inputs and print commands without execution |

## Pipeline

1. Clone VIPE repository
2. Setup virtual environment
3. Install dependencies
4. Prepare dataset (COLMAP format)
5. Run Gaussian Splatting training
6. Render demo video

## Requirements

- **NVIDIA GPU with CUDA** - required for training
- Dataset in COLMAP format or compatible structure

## Dependencies

```bash
pip install -e ".[task4-gpu]"
```

## Resources

- VIPE Project: https://github.com/nv-tlabs/vipe
- Dataset: [Google Drive](https://drive.google.com/drive/folders/1wPpU0irWLunZCCKR5TSk_bhofioQQ8eV)
