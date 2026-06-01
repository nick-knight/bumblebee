# On each clone of the repo, copy this script to local_setenv.sh, and modify it appropriately.

# (The local_setenv.sh copy does not get committed, so you'll want to version-control it by other means.)

# This script is meant to be sourced.

# By default, Hugging Face libraries cache things at ~/.cache/. This can cause surprises when working with large datasets.
# There are several ways to override HF caching behavior; the simplest seems to be setting HF_HOME.

export HF_HOME= # MANDATORY! (There is an assertion in train.py.)

# Various HF caches will default to subdirectories of HF_HOME. Ones controlled by environment variables include:
#   HF_DATASETS_CACHE  defaults to  $HF_HOME/datasets
#   HF_HUB_CACHE       defaults to  $HF_HOME/hub
#   HF_XET_CACHE       defaults to  $HF_HOME/xet
#   HF_ASSETS_CACHE    defaults to  $HF_HOME/assets
# Setting these variables overrides these defaults; Bumblebee assumes these variables are unset.

# Hugging Face Hub will rate-limit your downloads without passing a token.
export HF_TOKEN= # Optional, but recommended.

# By default, pip caches things in ~/.cache/. On a certain cluster I needed to override it.
export PIP_CACHE_DIR= # Optional.

# I have not implemented any Python Packaging so you'll have to pip-install whatever you need.
# When working with NVIDIA's PyTorch 25.08 container, the following suffices:
pip install datasets tiktoken --root-user-action ignore

# On a certain cluster, I get warnings that OpenMP is using its default of 1 thread. This suppresses that warning:
export OMP_NUM_THREADS=1
