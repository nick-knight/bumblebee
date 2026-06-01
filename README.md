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
