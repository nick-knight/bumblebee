from dataclasses import dataclass
import math
import torch

@dataclass
class ModelConfig:
    num_layers: int = 6
    hidden_dim: int = 512
    num_heads: int = 8
    num_query_groups: int = 8
    qk_dim: int = 64
    v_dim: int = 64
    mlp_dim: int = 2048
    max_seq_len: int = 512 # We override this elsewhere, based on TrainConfig.seq_len
    vocab_size: int = 1024 # We override this elsewhere, based on TrainConfig.{dataset, tokenizer_model}
    dtype: torch.dtype = torch.float32 # We override this elsewhere, based on device.type

class GPTModel(torch.nn.Module):

    def __init__ (self, model_config):

        mc = model_config

        assert mc.num_heads % mc.num_query_groups == 0, "GQA failure: num_query_groups must divide num_heads."

        super().__init__()

        # Dimensions are chosen so that the last corresponds to the matmul inner dimension.
        self.W_FC1 = torch.nn.Parameter(torch.empty((mc.num_layers, mc.mlp_dim, mc.hidden_dim), dtype=mc.dtype))
        self.W_FC2 = torch.nn.Parameter(torch.empty((mc.num_layers, mc.hidden_dim, mc.mlp_dim), dtype=mc.dtype))

        self.W_E   = torch.nn.Parameter(torch.empty((mc.hidden_dim, mc.vocab_size), dtype=mc.dtype))
        self.W_U   = torch.nn.Parameter(torch.empty((mc.vocab_size, mc.hidden_dim), dtype=mc.dtype))

        self.W_Q   = torch.nn.Parameter(torch.empty((mc.num_layers, mc.num_heads, mc.qk_dim, mc.hidden_dim), dtype=mc.dtype))
        self.W_K   = torch.nn.Parameter(torch.empty((mc.num_layers, mc.num_query_groups, mc.qk_dim, mc.hidden_dim), dtype=mc.dtype))
        self.W_V   = torch.nn.Parameter(torch.empty((mc.num_layers, mc.num_query_groups, mc.v_dim, mc.hidden_dim), dtype=mc.dtype))
        self.W_O   = torch.nn.Parameter(torch.empty((mc.num_layers, mc.num_heads, mc.hidden_dim, mc.v_dim), dtype=mc.dtype))

        # For a linear operator from R^N to R^M, pick matrix entries as i.i.d. samples from Uniform[-1/sqrt(N), 1/sqrt(N)].
        @torch.no_grad()
        def init_linear (W, N):
            std = 1. / math.sqrt(N)
            W.uniform_(-std, std)

        init_linear(self.W_E,   self.W_E.shape[-1])
        init_linear(self.W_U,   self.W_U.shape[-1])

        init_linear(self.W_FC1, self.W_FC1.shape[-1])
        init_linear(self.W_FC2, self.W_FC2.shape[-1])

        init_linear(self.W_Q,   self.W_Q.shape[-1])
        init_linear(self.W_K,   self.W_K.shape[-1])
        init_linear(self.W_V,   self.W_V.shape[-1])
        init_linear(self.W_O,   self.W_O.shape[-1])

        tok_idxs = torch.arange(mc.max_seq_len)
        # attn_mask is n_keys-by-n_queries, with True => ignore entry.
        # For causality, this means an upper triangular False matrix.
        self.register_buffer('attn_mask', tok_idxs[:, None] > tok_idxs[None, :], persistent=False)

        self.mc = model_config

    def forward(self, inputs):
        assert inputs.ndim == 2, "For simplicity, we require inputs is a 2D tensor, shape (B, S)."
        seq_len = inputs.size(1)
        assert seq_len <= self.mc.max_seq_len, "Input seq_len exceeds model's max_seq_len."

        X = self.W_E[:, inputs]
        X = torch.einsum('DBS->BSD', X)
        for layer in range(self.mc.num_layers):
            Z = torch.nn.functional.rms_norm(X, (X.size(-1),))

            Q = torch.einsum('HdD,BSD->HBSd', self.W_Q[layer, ...], Z)
            K = torch.einsum('HdD,BSD->HBSd', self.W_K[layer, ...], Z)
            V = torch.einsum('HdD,BSD->HBSd', self.W_V[layer, ...], Z)

            Z = torch.einsum('HBTd,HBSd->HBTS', K.repeat_interleave(self.mc.num_heads // self.mc.num_query_groups, dim=0), Q)
            Z = Z.div(math.sqrt(self.mc.qk_dim))
            Z = Z.masked_fill(self.attn_mask[:seq_len, :seq_len], float('-inf'))
            Z = torch.nn.functional.softmax(Z, dim=2)
            Z = torch.einsum('HBTd,HBTS->HBSd', V.repeat_interleave(self.mc.num_heads // self.mc.num_query_groups, dim=0), Z)

            Z = torch.einsum('HDd,HBSd->BSD', self.W_O[layer, ...], Z)

            X = X + Z
            Z = torch.nn.functional.rms_norm(X, (X.size(-1),))
            Z = torch.einsum('QD,BSD->BSQ', self.W_FC1[layer, ...], Z)
            Z = torch.nn.functional.relu(Z).square()

            Z = torch.einsum('DQ,BSQ->BSD', self.W_FC2[layer, ...], Z)
            X = X + Z
        X = torch.einsum('VD,BSD->BSV', self.W_U, X)

        return X
