import dataclasses
import os
from pathlib import Path

import torch


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    next_step,
    model_config,
    train_config,
    optim_config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    checkpoint = {
        'version': CHECKPOINT_VERSION,
        'next_step': next_step,
        'model_config': dataclasses.asdict(model_config),
        'train_config': dataclasses.asdict(train_config),
        'optim_config': dataclasses.asdict(optim_config),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path, *, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get('version') != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {checkpoint.get('version')}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
