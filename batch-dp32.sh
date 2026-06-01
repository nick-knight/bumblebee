#!/bin/bash

# This script is specific to my environment, on a specific internal cluster.
# It will not work out-of-the-box for anyone else. I'm just sharing it here as a reference.
# I have suppressed any sensitive details. Colleagues: contact me via internal channels for help.

# I adapt this script for "production" runs. (Please see interactive-dp8.sh for interactive development/testing.)

#SBATCH --job-name=bumblebee
#SBATCH --time=04:00:00
#SBATCH --account=adlr_psx_numerics
#SBATCH --partition=batch_block1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8

# NKLFS stands for Nick Knight Lustre File System. I set it in my .bashrc to point to my top-level development directory on /scratch.
: "${NKLFS:?Error: NKLFS must be set!}"

CONTAINER="$NKLFS/containers/25.08-py3.sqsh"
WORKDIR="$NKLFS/bumblebee"

# It does not appear to be guaranteed that $MASTER_ADDR will equal the node where this script
# is being run. This shouldn't matter, since the Rendezvous endpoint can be chosen arbitrarily.
MASTER_ADDR=$(scontrol show hostnames | head -n 1)

# This has to be unique; there isn't an easy way to ensure this so we just pick a random port.
# If it errors, then we need to pick a different port.
MASTER_PORT=54321

# Note 1: some docs suggest that the --container-* Pyxis arguments can be passed via sbatch directive,
# but this didn't work (on a certain cluster anyway).
# Note 2: the backslash escape on SLURM_NODEID is crucial. The escape on SLURM_GPUS_ON_NODE doesn't
# matter in the (common) case that all allocated nodes have the same number of GPUs.
srun \
    --container-image="$CONTAINER" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="$WORKDIR" \
    bash << EOF
. local_setenv.sh
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=\$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --node_rank=\$SLURM_NODEID \
    train.py
EOF
