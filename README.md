# UConn Senior Design Project - Team 19

## Setup

### Environment

```bash
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Start virtual environment
uv env
# 3. Install dependencies
uv pip install -r requirements.txt
```

### Hugging Face / Dataset

To generate a Personal Access Token (PAT), create a new token with write permissions: https://huggingface.co/settings/tokens

```bash
# 1. Ensure hugging face CLI is installed
hf
# 2. Login to hugging face
hf auth login # should be prompted to enter your PAT
# 3. Download dataset
hf download jobegets/ev-fsae-testing --repo-type dataset --local-dir ./test-data
```
