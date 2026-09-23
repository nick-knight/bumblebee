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

### Local toy training

The `modelopt` branch can train from a local UTF-8 text file without loading Hugging Face Datasets.
It uses the first 90% of tokens for training and the remainder for validation. For example:

```
python train.py \
    --dataset-name text \
    --dataset-path /path/to/corpus.txt \
    --train-steps 1000 \
    --seq-len 128 \
    --num-layers 4 \
    --hidden-dim 256 \
    --num-heads 4 \
    --num-query-groups 4 \
    --qk-dim 64 \
    --v-dim 64 \
    --mlp-dim 1024 \
    --checkpoint-path checkpoints/parent.pt \
    --checkpoint-interval 100
```

Resume training with the same model and training arguments plus
`--resume-from checkpoints/parent.pt`. Checkpoints contain the model configuration, training
configuration, model weights, optimizer state, and next training step. Data-loader position is not
captured, so resumed runs are not bitwise identical to uninterrupted runs.

The model also exposes `set_key_value_transforms()`. ModelOpt experiments can use it to install
same-shaped calibration or projection modules immediately after each layer computes its keys and
values. Identity modules are installed by default, preserving the original model behavior.

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
