import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import tiktoken

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from model import ModelConfig, GPTModel
from dataloader import dataloader, synthetic_dataloader

@dataclass
class TrainConfig:
    dataset_name: str = 'fineweb'
    tokenizer_model: str = 'gpt2'
    train_local_microbatch_size: int = 8
    valid_local_microbatch_size: int = 8
    seq_len: int = 512
    ignore_index: int = -1
    train_steps: int = 1000
    valid_interval: int = 10
    num_train_microbatches: int = 1
    num_valid_microbatches: int = 1
    lr: float = 1e-4
    torch_compile: bool = True

def train_step(
        model,
        dataloader,
        optimizer,
        num_microbatches,
        ignore_index,
        is_distributed,
        device
    ):

    model.train()
    model.zero_grad(set_to_none=True)

    loss_accum = torch.zeros((), dtype=torch.float32, device=device)
    ntok_accum = torch.zeros((), dtype=torch.int64, device=device)

    for _ in range(num_microbatches):
        local_microbatch = next(dataloader)
        outputs = model(local_microbatch)
        targets = local_microbatch.roll(-1, 1)
        targets[:, -1] = ignore_index
        loss = torch.nn.functional.cross_entropy(
            outputs.to(torch.float32).view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction='sum',
        )
        del outputs   # Hack to enable Python garbage collector to reclaim this memory during backward().
        loss_accum += loss
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

    optimizer.step()

    # Only correct on rank 0:
    return loss_accum.div_(ntok_accum).item()

def valid_step(
    model,
    dataloader,
    num_microbatches,
    ignore_index,
    is_distributed,
    device,
):

    model.eval()

    loss_accum = torch.zeros((), dtype=torch.float32, device=device)
    ntok_accum = torch.zeros((), dtype=torch.int64, device=device)

    for _ in range(num_microbatches):
        local_microbatch = next(dataloader)
        with torch.inference_mode():
            outputs = model(local_microbatch)
        targets = local_microbatch.roll(-1, 1)
        targets[:, -1] = ignore_index
        loss_accum += torch.nn.functional.cross_entropy(
            outputs.to(torch.float32).view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=ignore_index,
            reduction='sum',
        )
        ntok_accum += targets.size(0) * (targets.size(1) - 1)

    if is_distributed:
        dist.reduce(loss_accum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(ntok_accum, dst=0, op=dist.ReduceOp.SUM)

    # Only correct on rank 0:
    return loss_accum.div_(ntok_accum).item()

def run_training(
        model_config,
        train_config,
        train_ds = None,
        valid_ds = None,
        seed: int = 666,
    ):
    rank, world_size, is_distributed, device = setup_runtime()
    _, print0 = make_print_helpers(rank)

    tc = train_config
    mc = model_config

    # We need to patch up mc.max_seq_len, mc.vocab_size, and mc.dtype.

    mc.max_seq_len = tc.seq_len

    if tc.dataset_name == 'synthetic':
        mc.vocab_size = 512 # small value for smoke-testing
    else:
        mc.vocab_size = tiktoken.encoding_for_model(tc.tokenizer_model).n_vocab

    mc.dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    if tc.dataset_name == 'synthetic':
        train_dataloader = synthetic_dataloader(mc.vocab_size, tc.train_local_microbatch_size, tc.seq_len, device, seed=seed + rank)
        valid_dataloader = synthetic_dataloader(mc.vocab_size, tc.valid_local_microbatch_size, tc.seq_len, device, seed=seed + world_size + rank)
    elif tc.dataset_name == 'fineweb':
        # For sample-10BT, Hugging Face Hub and Datasets consume 76 GiB on my setup.
        assert os.getenv('HF_HOME'), "Please set HF_HOME to a place you're OK with Hugging Face caching a bunch of stuff."
        ds = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT', split='train', streaming=False)
        ds = ds.shuffle(seed)
        if is_distributed:
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        ds = ds.train_test_split(test_size=0.1, shuffle=False)
        tokenizer = tiktoken.get_encoding(tc.tokenizer_model)
        train_dataloader = dataloader(ds['train'], tokenizer, tc.train_local_microbatch_size, tc.seq_len, device)
        valid_dataloader = dataloader(ds['test'],  tokenizer, tc.valid_local_microbatch_size, tc.seq_len, device)

    torch.manual_seed(seed)

    model = GPTModel(mc).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr)

    if tc.torch_compile:
        model = torch.compile(model)

    for step in range(tc.train_steps):

        loss = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            num_microbatches=tc.num_train_microbatches,
            ignore_index=tc.ignore_index,
            is_distributed=is_distributed,
            device=device
        )

        print0(f"After training step {step}, training loss is {loss} nats.")

        if step % tc.valid_interval == 0:
            loss = valid_step(
                model=model,
                dataloader=valid_dataloader,
                num_microbatches=tc.num_valid_microbatches,
                ignore_index=tc.ignore_index,
                is_distributed=is_distributed,
                device=device
            )

            print0(f"After training step {step}, validation loss is {loss} nats.")

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
    # This reference implementation does not actually check environment or command-line. Chat will be happy to add this for you.
    model_config = ModelConfig()
    train_config = TrainConfig()



    return model_config, train_config

def main():
    model_config, train_config = make_configs()
    run_training(model_config, train_config)

if __name__ == '__main__':
    main()
