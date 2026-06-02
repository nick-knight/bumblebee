<p align="center">
  <img src="assets/bumblebee.png" alt="Bumblebee" width="400">
</p>

# Bumblebee

[Bumblebee](https://en.wikipedia.org/wiki/Bumblebee_(Transformers)) is a small Transformer.

## Project setup:

1. Clone repo.

2. `cp setenv.sh local_setenv.sh`

    Configure `local_setenv.sh` as detailed in that file.

## Basic training example (single worker):

1. `. local_setenv.sh`

2. `python train.py`

The first time you run `train.py`, it will download and preprocess a very large Hugging Face dataset into `$HF_HOME`. This may take hours.

## Cluster usage

See `interactive-dp8.sh` and `batch-dp32.sh`.

## Development

Testing: currently there is no automated testing. I manually test, from time to time, by running `python train.py` and declaring success when the loss looks like it's going down.

I use a pre-commit Git hook that calls ruff-check. To install this hook on your clone, run the following from the top-level directory:

```
# Install pre-commit and ruff, if you don't have them already:
pip install pre-commit ruff

# Install the hook, specified in .pre-commit-config.yaml, to .git/hooks/pre-commit
pre-commit install

# Ensure that it installed correctly.
pre-commit run --all-files
```
