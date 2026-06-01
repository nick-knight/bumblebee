# This script is specific to my environment, on a specific internal cluster.
# It will not work out-of-the-box for anyone else. I'm just sharing it here as a reference.
# I have suppressed any sensitive details. Colleagues: contact me via internal channels for help.

# I adapt this script for interactive development and smoke-testing. (Please see batch-dp32.sh for "production" runs.)

# NKLFS stands for Nick Knight Lustre File System. I set it in my .bashrc to point to my top-level development directory on /scratch.
: "${NKLFS:?Error: NKLFS must be set!}"

CONTAINER="$NKLFS/containers/25.08-py3.sqsh"
WORKDIR="$NKLFS/bumblebee"

srun \
	--job-name=bumblebee \
	--time=04:00:00 \
	--account=adlr_psx_numerics \
	--partition=interactive \
	--nodes=1 \
	--ntasks-per-node=1 \
	--gpus-per-task=8 \
	--container-image="$CONTAINER" \
	--container-mounts /lustre:/lustre \
	--container-workdir="$WORKDIR" \
	--pty bash

# Once you get the interactive node,
# . local_setenv.sh # see instructions in setenv.sh
# . torchrun --standalone --nproc-per-node=8 train.py

# For multi-node, you'll need a more complicated torchrun command, which can be deduced from batch-dp32.sh.
