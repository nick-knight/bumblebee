import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from checkpoint import load_checkpoint, save_checkpoint
from dataloader import dataloader, get_tokenizer, synthetic_dataloader, text_dataloader
from model import ModelConfig, GPTModel
from optimizer import OptimConfig, Optimizer

@dataclass
class TrainConfig:
    dataset_name: str = 'fineweb'
    dataset_path: str | None = None
    tokenizer_model: str = 'gpt2'
    train_local_microbatch_size: int = 8
    valid_local_microbatch_size: int = 8
    seq_len: int = 512
    ignore_index: int = -1
    train_steps: int = 1000
    log_interval: int = 1
    valid_interval: int = 10
    num_train_microbatches: int = 1
    num_valid_microbatches: int = 1
    torch_compile: bool = True
    checkpoint_path: str | None = None
    checkpoint_interval: int | None = None
    resume_from: str | None = None

def train_step(
        model,
        dataloader,
        optimizer,
        step,
        num_microbatches,
        ignore_index,
        is_distributed
    ):

    model.train()
    model.zero_grad(set_to_none=True)

    device = next(model.parameters()).device

    loss_dtype = torch.float32
    loss_accum_dtype = torch.float64 if device.type != 'mps' else torch.float32 # MPS doesn't support FP64 yet.

    loss_accum = torch.zeros((), dtype=loss_accum_dtype, device=device)
    ntok_accum = torch.zeros((), dtype=torch.int64, device=device)

    for _ in range(num_microbatches):
        local_microbatch = next(dataloader)
        outputs = model(local_microbatch)
        targets = local_microbatch.roll(-1, 1)
        targets[:, -1] = ignore_index
        loss = torch.nn.functional.cross_entropy(
            outputs.to(loss_dtype).view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction='sum',
        )
        del outputs   # Hack to enable Python garbage collector to reclaim this memory during backward().
        loss_accum += loss.detach().to(loss_accum_dtype)
        loss.backward()
        ntok_accum += targets.size(0) * (targets.size(1) - 1)

    if is_distributed:
        dist.all_reduce(ntok_accum)
        # Pack grads into one all-reduce. Nontrivial memory cost for a nontrivial speedup.
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat = torch.cat([g.view(-1) for g in grads])
        dist.all_reduce(flat)
        flat.div_(ntok_accum)
        offset = 0
        for g in grads:
            g.copy_(flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()
        dist.reduce(loss_accum, dst=0, op=dist.ReduceOp.SUM)
    else:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(ntok_accum)

    # Pass step as a tensor to enable torch.compile.
    optimizer.step(torch.tensor(step, dtype=torch.int64, device=device))

    # Only correct on rank 0:
    return loss_accum.div_(ntok_accum).item()

def valid_step(
    model,
    dataloader,
    num_microbatches,
    ignore_index,
    is_distributed
):

    model.eval()

    device = next(model.parameters()).device

    loss_dtype = torch.float32
    loss_accum_dtype = torch.float64 if device.type != 'mps' else torch.float32 # MPS doesn't support FP64 yet.

    loss_accum = torch.zeros((), dtype=loss_accum_dtype, device=device)
    ntok_accum = torch.zeros((), dtype=torch.int64, device=device)

    for _ in range(num_microbatches):
        local_microbatch = next(dataloader)
        with torch.inference_mode():
            outputs = model(local_microbatch)
        targets = local_microbatch.roll(-1, 1)
        targets[:, -1] = ignore_index
        loss_accum += torch.nn.functional.cross_entropy(
            outputs.to(loss_dtype).view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction='sum',
        ).to(loss_accum_dtype)
        ntok_accum += targets.size(0) * (targets.size(1) - 1)

    if is_distributed:
        dist.reduce(loss_accum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(ntok_accum, dst=0, op=dist.ReduceOp.SUM)

    # Only correct on rank 0:
    return loss_accum.div_(ntok_accum).item()

def run_training(
        model_config,
        train_config,
        optim_config,
        seed: int = 666,
    ):
    rank, world_size, is_distributed, device = setup_runtime()
    _, print0 = make_print_helpers(rank)

    mc = model_config
    tc = train_config
    oc = optim_config

    if tc.log_interval < 1:
        raise ValueError("log_interval must be positive")
    if tc.checkpoint_interval is not None and tc.checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if tc.dataset_name == 'text' and tc.dataset_path is not None:
        tc.dataset_path = str(Path(tc.dataset_path).expanduser().resolve())

    # We need to patch up mc.max_seq_len, mc.vocab_size, and mc.dtype.
    mc.max_seq_len = tc.seq_len

    if tc.dataset_name == 'synthetic':
        mc.vocab_size = 512 # small value for smoke-testing
    elif tc.dataset_name in ('fineweb', 'text'):
        mc.vocab_size = get_tokenizer(tc.tokenizer_model).n_vocab
    else:
        raise ValueError(f"Unsupported dataset_name: {tc.dataset_name}")

    # Make model fprop/bprop use BF16 on CUDA/MPS, and FP32 otherwise:
    if device.type == 'cuda' or device.type == 'mps':
        mc.narrow_dtype = torch.bfloat16
    else:
        mc.narrow_dtype = torch.float32

    if tc.dataset_name == 'synthetic':
        train_dataloader = synthetic_dataloader(mc.vocab_size, tc.train_local_microbatch_size, tc.seq_len, device, seed=seed + rank)
        valid_dataloader = synthetic_dataloader(mc.vocab_size, tc.valid_local_microbatch_size, tc.seq_len, device, seed=seed + world_size + rank)
    elif tc.dataset_name == 'text':
        if tc.dataset_path is None:
            raise ValueError("dataset_path is required when dataset_name='text'")
        tokenizer = get_tokenizer(tc.tokenizer_model)
        train_dataloader = text_dataloader(
            tc.dataset_path,
            tokenizer,
            tc.train_local_microbatch_size,
            tc.seq_len,
            device,
            seed + rank,
            'train',
        )
        valid_dataloader = text_dataloader(
            tc.dataset_path,
            tokenizer,
            tc.valid_local_microbatch_size,
            tc.seq_len,
            device,
            seed + world_size + rank,
            'valid',
        )
    elif tc.dataset_name == 'fineweb':
        # Keep Hugging Face dependencies optional for synthetic and local-text experiments.
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        # For sample-10BT, Hugging Face Hub and Datasets consume 76 GiB on my setup.
        assert os.getenv('HF_HOME'), "Please set HF_HOME to a place you're OK with Hugging Face caching a bunch of stuff."
        ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=False)
        ds = ds.shuffle(seed)
        if is_distributed:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        ds = ds.train_test_split(test_size=0.1, shuffle=False)
        tokenizer = get_tokenizer(tc.tokenizer_model)
        train_dataloader = dataloader(ds['train'], tokenizer, tc.train_local_microbatch_size, tc.seq_len, device)
        valid_dataloader = dataloader(ds['test'],  tokenizer, tc.valid_local_microbatch_size, tc.seq_len, device)

    torch.manual_seed(seed)

    model = GPTModel(mc).to(device)

    # Ensure the GPTModel has been instantiated and moved to device before instantiating the Optimizer
    optimizer = Optimizer(oc, model)

    start_step = 0
    if tc.resume_from is not None:
        checkpoint = load_checkpoint(tc.resume_from, model=model, optimizer=optimizer, map_location=device)
        start_step = checkpoint['next_step']

    checkpoint_model = model

    if tc.torch_compile:
        # The 'cpu' backend seems to have a lot of issues.
        if device.type == 'cuda' or device.type == 'mps':
            model = torch.compile(model)

        # This works, but is fragile, and has dubious speedups. Leaving it disabled.
        # optimizer.step = torch.compile(optimizer.step)

    for step in range(start_step, tc.train_steps):

        loss = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            step=step,
            num_microbatches=tc.num_train_microbatches,
            ignore_index=tc.ignore_index,
            is_distributed=is_distributed
        )

        if step % tc.log_interval == 0:
            print0(f"After training step {step}, training loss is {loss} nats.")

        if step % tc.valid_interval == 0:
            loss = valid_step(
                model=model,
                dataloader=valid_dataloader,
                num_microbatches=tc.num_valid_microbatches,
                ignore_index=tc.ignore_index,
                is_distributed=is_distributed
            )

            print0(f"After training step {step}, validation loss is {loss} nats.")

        if (
            rank == 0
            and tc.checkpoint_path is not None
            and tc.checkpoint_interval is not None
            and (step + 1) % tc.checkpoint_interval == 0
        ):
            save_checkpoint(
                tc.checkpoint_path,
                model=checkpoint_model,
                optimizer=optimizer,
                next_step=step + 1,
                model_config=mc,
                train_config=tc,
                optim_config=oc,
            )

    if rank == 0 and tc.checkpoint_path is not None:
        save_checkpoint(
            tc.checkpoint_path,
            model=checkpoint_model,
            optimizer=optimizer,
            next_step=tc.train_steps,
            model_config=mc,
            train_config=tc,
            optim_config=oc,
        )

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

def setup_runtime():
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if is_distributed:
        backend = 'nccl' if device.type == 'cuda' else 'gloo'
        dist.init_process_group(backend=backend)

    return rank, world_size, is_distributed, device

def make_print_helpers(rank):
    def print_n(message):
        print(f"[{rank}]: {message}")

    def print_0(message):
        if rank == 0:
            print_n(message)

    return print_n, print_0

def make_configs():
    model_config = ModelConfig()
    train_config = TrainConfig()
    optim_config = OptimConfig()

    return model_config, train_config, optim_config

def parse_configs():
    model_config, train_config, optim_config = make_configs()
    parser = argparse.ArgumentParser(description='Train the Bumblebee Transformer.')
    parser.add_argument('--dataset-name', choices=('fineweb', 'synthetic', 'text'))
    parser.add_argument('--dataset-path')
    parser.add_argument('--tokenizer-model')
    parser.add_argument('--train-steps', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--valid-interval', type=int)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--train-local-microbatch-size', type=int)
    parser.add_argument('--valid-local-microbatch-size', type=int)
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--resume-from')
    parser.add_argument('--torch-compile', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--hidden-dim', type=int)
    parser.add_argument('--num-heads', type=int)
    parser.add_argument('--num-query-groups', type=int)
    parser.add_argument('--qk-dim', type=int)
    parser.add_argument('--v-dim', type=int)
    parser.add_argument('--mlp-dim', type=int)
    parser.add_argument('--max-lr', type=float)
    parser.add_argument('--max-wd', type=float)
    args = vars(parser.parse_args())

    for name in (
        'dataset_name',
        'dataset_path',
        'tokenizer_model',
        'train_steps',
        'log_interval',
        'valid_interval',
        'seq_len',
        'train_local_microbatch_size',
        'valid_local_microbatch_size',
        'checkpoint_path',
        'checkpoint_interval',
        'resume_from',
        'torch_compile',
    ):
        if args[name] is not None:
            setattr(train_config, name, args[name])
    for name in (
        'num_layers',
        'hidden_dim',
        'num_heads',
        'num_query_groups',
        'qk_dim',
        'v_dim',
        'mlp_dim',
    ):
        if args[name] is not None:
            setattr(model_config, name, args[name])
    for name in ('max_lr', 'max_wd'):
        if args[name] is not None:
            setattr(optim_config, name, args[name])

    return model_config, train_config, optim_config

def main():
    model_config, train_config, optim_config = parse_configs()
    run_training(model_config, train_config, optim_config)

if __name__ == '__main__':
    main()
