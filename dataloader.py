import torch

def dataloader(dataset, tokenizer, batch_size, seq_len, device):

    eot = tokenizer._special_tokens['<|endoftext|>']

    ds_it = iter(dataset)

    if device.type == 'cuda':
        inputs_cpu_buf = torch.empty((batch_size*seq_len,), dtype=torch.int64, pin_memory=True)
        inputs_gpu = torch.empty((batch_size, seq_len), dtype=torch.int64, device=device)
    else:
        inputs_cpu_buf = torch.empty((batch_size*seq_len,), dtype=torch.int64)

    while True:
        pos = 0

        # Sequence packing strategy:
        # Pack successive tokenized documents, each with EOS prepended, into the buffer.
        # If a document overflows the buffer, truncate it and discard its remainder.
        while pos < batch_size * seq_len:
            doc = next(ds_it)['text']
            tokens = torch.tensor([eot] + tokenizer.encode_ordinary(doc), dtype=torch.int64)
            num_to_take = min(batch_size * seq_len - pos, tokens.size(0))
            inputs_cpu_buf[pos:pos + num_to_take] = tokens[:num_to_take]
            pos += num_to_take

        if device.type == 'cuda':
            inputs_gpu.copy_(inputs_cpu_buf.view(batch_size, seq_len), non_blocking=True)
            yield inputs_gpu
        else:
            yield inputs_cpu_buf.view(batch_size, seq_len).to(device)

# Useful for smoke-testing. (Hugging Face Datasets has painful startup time, even just validating its cache.)
def synthetic_dataloader(vocab_size, batch_size, seq_len, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    while True:
        yield torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
