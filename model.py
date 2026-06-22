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
    narrow_dtype: torch.dtype = torch.float32 # We override this elsewhere, based on device.type

class GPTModel(torch.nn.Module):

    def __init__ (self, model_config):

        self.num_layers = model_config.num_layers
        self.hidden_dim = model_config.hidden_dim
        self.num_heads = model_config.num_heads
        self.num_query_groups = model_config.num_query_groups
        self.qk_dim = model_config.qk_dim
        self.v_dim = model_config.v_dim
        self.mlp_dim = model_config.mlp_dim
        self.max_seq_len = model_config.max_seq_len
        self.vocab_size = model_config.vocab_size
        self.narrow_dtype = model_config.narrow_dtype

        # Not currently exposed via ModelConfig:
        self.wide_dtype = torch.float32

        assert self.num_heads % self.num_query_groups == 0, "GQA failure: num_query_groups must divide num_heads."

        super().__init__()

        # Dimensions are ordered so that the last corresponds to the matmul inner dimension.
        self.W_E   = torch.nn.Parameter(torch.empty((self.hidden_dim, self.vocab_size), dtype=self.wide_dtype))
        self.W_U   = torch.nn.Parameter(torch.empty((self.vocab_size, self.hidden_dim), dtype=self.wide_dtype))

        self.W_Q   = torch.nn.Parameter(torch.empty((self.num_layers, self.num_heads, self.qk_dim, self.hidden_dim), dtype=self.wide_dtype))
        self.W_K   = torch.nn.Parameter(torch.empty((self.num_layers, self.num_query_groups, self.qk_dim, self.hidden_dim), dtype=self.wide_dtype))
        self.W_V   = torch.nn.Parameter(torch.empty((self.num_layers, self.num_query_groups, self.v_dim, self.hidden_dim), dtype=self.wide_dtype))
        self.W_O   = torch.nn.Parameter(torch.empty((self.num_layers, self.num_heads, self.hidden_dim, self.v_dim), dtype=self.wide_dtype))

        self.W_FC1 = torch.nn.Parameter(torch.empty((self.num_layers, self.mlp_dim, self.hidden_dim), dtype=self.wide_dtype))
        self.W_FC2 = torch.nn.Parameter(torch.empty((self.num_layers, self.hidden_dim, self.mlp_dim), dtype=self.wide_dtype))

        # GPT-2-style initialization: all weights ~ Normal(0, 0.02), with W_FC2 and W_O additionally scaled by 1/sqrt(2 * num_layers).
        @torch.no_grad()
        def init_normal (W, std):
            W.normal_(mean=0., std=std)

        residual_scale = 1. / math.sqrt(2 * self.num_layers)

        init_normal(self.W_E,   0.02)
        init_normal(self.W_U,   0.02)

        init_normal(self.W_Q,   0.02)
        init_normal(self.W_K,   0.02)
        init_normal(self.W_V,   0.02)
        init_normal(self.W_O,   0.02 * residual_scale)

        init_normal(self.W_FC1, 0.02)
        init_normal(self.W_FC2, 0.02 * residual_scale)

        tok_idxs = torch.arange(self.max_seq_len)
        # attn_mask is n_keys-by-n_queries, with True => ignore entry.
        # For causality, this means an upper triangular False matrix.
        self.register_buffer('attn_mask', tok_idxs[:, None] > tok_idxs[None, :], persistent=False)

    def forward(self, inputs):
        assert inputs.ndim == 2, "For simplicity, we require inputs is a 2D tensor, shape (B, S)."
        seq_len = inputs.size(1)
        assert seq_len <= self.max_seq_len, "Input seq_len exceeds model's max_seq_len."

        # Mixed-precision recipe:
        #  * wide arithmetic for un/embedding, softmax, norms, and residuals.
        #  * narrow arithmetic for QKV, BMM1, BMM2, Proj, and FFN.

        nd = self.narrow_dtype
        wd = self.wide_dtype

        X = self.W_E[:, inputs]
        X = torch.einsum('DBS->BSD', X)

        for layer in range(self.num_layers):

            Z = torch.nn.functional.rms_norm(X, (X.size(-1),))
            Q = torch.einsum('HdD,BSD->HBSd', self.W_Q[layer, ...].to(nd), Z.to(nd))
            K = torch.einsum('HdD,BSD->HBSd', self.W_K[layer, ...].to(nd), Z.to(nd))
            V = torch.einsum('HdD,BSD->HBSd', self.W_V[layer, ...].to(nd), Z.to(nd))
            Z = torch.einsum('HBTd,HBSd->HBTS', K.repeat_interleave(self.num_heads // self.num_query_groups, dim=0), Q)
            Z = Z.div(math.sqrt(self.qk_dim))
            Z = Z.masked_fill(self.attn_mask[:seq_len, :seq_len], float('-inf'))
            Z = torch.nn.functional.softmax(Z.to(wd), dim=2)
            Z = torch.einsum('HBTd,HBTS->HBSd', V.repeat_interleave(self.num_heads // self.num_query_groups, dim=0), Z.to(nd))
            Z = torch.einsum('HDd,HBSd->BSD', self.W_O[layer, ...].to(nd), Z)
            X = X + Z.to(wd)

            Z = torch.nn.functional.rms_norm(X, (X.size(-1),))
            Z = torch.einsum('QD,BSD->BSQ', self.W_FC1[layer, ...].to(nd), Z.to(nd))
            Z = torch.nn.functional.relu(Z).square()
            Z = torch.einsum('DQ,BSQ->BSD', self.W_FC2[layer, ...].to(nd), Z)
            X = X + Z.to(wd)

        X = torch.einsum('VD,BSD->BSV', self.W_U, X)

        return X
