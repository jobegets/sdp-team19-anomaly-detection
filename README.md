# UConn Senior Design Project - Team 19

## Setup

### Environment

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Start virtual environment (optional if already created)
uv venv
# 3. Activate virtual environment
source .venv/bin/activate
# 3. Install dependencies
uv pip install -r requirements.txt
```

### Hugging Face / Dataset

To generate a Personal Access Token (PAT), create a new token with write permissions: https://huggingface.co/settings/tokens

```bash
# 1. Ensure hugging face CLI is installed
hf
# 1.1 If not:
curl -LsSf https://hf.co/cli/install.sh | bash
# 2. Login to hugging face
hf auth login # should be prompted to enter your PAT
# 3. Download dataset
hf download jobegets/ev-fsae-testing --repo-type dataset --local-dir ./test-data
```

### Deployment

https://gweb-coral-full.uc.r.appspot.com/docs/dev-board/get-started/#connect-via-mdt

1. Ensure connection to board via `mdt devices`
2. Open shell on board `mdt shell`
3. Clone this repo: `git clone https://github.com/jobegets/sdp-team19-anomaly-detection.git`
4. ^ Follow steps above. Setup venv and huggingface.
   4.1. Might need to install newer python version. You can do this easily with uv (while in venv)! `uv python install 3.13`
5. Run application REPL: `uv run python -m src.main`
6. pray

### CLI

Use the centralized REPL entrypoint:

```bash
uv run python -m src.main
```

Menu options in the REPL:
1. Choose dataset (global selection)
2. Train model on selected dataset (always updates artifact)
3. Evaluate last trained model using its dataset
4. Compare all trained models from the current REPL session
5. Auto split train/test:
   - Train set: all `NO_FAULTS` datasets
   - Test set: all `HAS_FAULTS` datasets
6. Exit

Dataset organization behavior:
- Non-moving samples are always ignored (`RPM <= 0.5`).
- Dataset picker shows labels (`HAS_FAULTS`, `NO_FAULTS`) and excludes datasets that are empty after filtering.
- A dataset manifest is written to `artifacts/dataset_manifest.csv` each session start.

### Debugging

After activating venv and attempting to install packages with pip:
`No module named 'pip'`

Try `python -m ensurepip --default-pip`
