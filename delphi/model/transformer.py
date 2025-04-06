"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from delphi.model.components import (
    AgeEncoding,
    DelphiEmbedding,
    ZeroTimeInflationPiProjector,
)


@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    zero_time_inflation: bool = False
    pi_projector: str = "linear"  # 'linear' or 'mlp'


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = False  # hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, attn_mask):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att


class Delphi(nn.Module):

    def __init__(self, config: DelphiConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                embed=DelphiEmbedding(config),
                # wte=nn.Embedding(config.vocab_size, config.n_embd),
                # wae=AgeEncoding(config),
                # token_drop=nn.Dropout(config.token_dropout),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.embed.token_embedding.weight = self.lm_head.weight

        # zero time inflation
        if config.zero_time_inflation:
            if config.pi_projector == "linear":
                self.pi_head = nn.Linear(config.vocab_size, 1, bias=False)
            elif config.pi_projector == "mlp":
                self.pi_head = ZeroTimeInflationPiProjector(config)
            else:
                raise ValueError(f"Unknown pi_projector: {config.pi_projector}")

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], torch.Tensor]:

        device = idx.device
        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # age_emb = self.transformer.wae(
        #     age.unsqueeze(-1)
        # )  # age embeddings of shape (b, t, n_embd)
        # x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        # x = x + age_emb

        x = self.transformer.embed(idx, age)  # (b, t, n_embd)
        x = self.transformer.drop(x)

        attn_mask = self.transformer.embed.attention_mask(
            idx, age, targets, targets_age
        )

        # attn_mask = (idx > 0).view(idx.size(0), 1, 1, idx.size(1)) * (idx > 0).view(
        #     idx.size(0), 1, idx.size(1), 1
        # )  # Do not attend to padded positions
        # attn_mask *= (
        #     torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[
        #         None, None, :, :
        #     ]
        #     > 0
        # )  # self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        # if targets is not None and self.config.mask_ties:
        #     attn_mask *= age.view(idx.size(0), 1, 1, idx.size(1)) != targets_age.view(
        #         idx.size(0), 1, idx.size(1), 1
        #     )  # Mask co-occuring tokens
        #     attn_mask += (attn_mask.sum(-1, keepdim=True) == 0) * torch.diag(
        #         torch.ones(idx.size(1), device=device)
        #     ) > 0 # todo: find out what this does
        # attn_mask = (
        #     attn_mask
        #     + (idx == 0).view(idx.size(0), 1, 1, idx.size(1))
        #     * torch.diag(torch.ones(idx.size(1), device=device))
        #     > 0
        # )  # Except for padding
        # attn_mask *= (
        #     torch.tril(torch.ones(idx.size(1), idx.size(1), device=device))[
        #         None, None, :, :
        #     ]
        #     > 0
        # )  # self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0

        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            # next token cross entropy loss, padding masked
            logits = self.lm_head(x)
            if self.config.zero_time_inflation:
                pi = self.pi_head(logits).squeeze()

            # if we are given some desired targets also calculate the loss
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[..., ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1
            for k in ignored_tokens:  # and gender
                pass_tokens *= targets != k

            # age_min = age.gather(1,(((idx >=4) * (idx <=12)) + 0).argmax(1)[:,None])
            # logits[...,-1][age <= age_min] = -100. #-float('Inf') ## Death can only occur after age_min

            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1))[pass_tokens],
                targets[pass_tokens],
                ignore_index=-1,
            )

            # time to next event loss, padding masked
            lse = torch.logsumexp(
                logits, -1
            )  ## More forgiving than using torch.max() for the most likely next event
            lse = -torch.log(torch.exp(-lse) + self.config.t_min)
            dt = targets_age - age
            if not self.config.zero_time_inflation:
                dt = torch.clamp(dt, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(
                    dt,
                    -1,
                    (
                        attn_mask
                        * torch.arange(
                            0, idx.size(1), device=device, dtype=torch.float32
                        ).view(1, 1, 1, -1)
                    )
                    .max(-1)
                    .indices.squeeze((1, 2)),
                )  # Use time from last untied token
            ldt = -torch.log(dt + torch.tensor(self.config.t_min))

            exp_log_likelihood = lse - torch.exp(lse - ldt)
            if not self.config.zero_time_inflation:
                loss_dt = -exp_log_likelihood.reshape(
                    -1
                )  ## Exponential log-likelihood (real statistics, TM)
            else:
                zero_case = -(F.softplus(-pi + lse) - F.softplus(-pi))
                nonzero_case = -(exp_log_likelihood - pi - F.softplus(-pi))
                loss_dt = (zero_case * (dt == 0) + nonzero_case * (dt > 0)).reshape(-1)
            loss_dt = torch.mean(loss_dt[pass_tokens])

            # Both losses combined
            # loss = loss_ce + loss_dt
            loss = {"loss_ce": loss_ce, "loss_dt": loss_dt}
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, :, :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, att

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurrence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        max_new_tokens: int = 100,
        max_age: float = 85 * 365.25,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        no_repeat: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if max_new_tokens == -1:
            max_new_tokens = 10000
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx  # if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            age_cond = age  # if age.size(1) <= self.config.block_size else age[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond, age_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            logits[:, self.config.ignore_tokens] = -float("Inf")
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if no_repeat:
                fill = idx + 0
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -float("Inf"))

            # apply softmax to convert logits to (normalized) probabilities
            # probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1)
            # lse = torch.logsumexp(logits, -1)[:,None]
            t_next = torch.clamp(
                -torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(),
                min=0,
                max=365 * 80.0,
            ).min(1)
            # age_next = age[...,[-1]] + torch.clamp(-torch.exp(-lse) * torch.rand(lse.shape, device=idx.device).log(), min=self.config.t_min, max=365*80.) #torch.normal(torch.zeros((1,1), device=idx.device),1.)
            idx_next = t_next[1][:, None]
            age_next = age[..., [-1]] + t_next[0][:, None]

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            if torch.all(idx_next == self.config.vocab_size - 1) or torch.all(
                age_next > max_age
            ):
                break

        pad = (
            torch.cumsum(torch.cumsum(idx == self.config.vocab_size - 1, 1).int(), 1)
            > 1
        ) + (age > max_age)
        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = float("NaN")
        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack(
                [
                    logits[:, j].scatter_(1, fill[:, : j + 1], float("NaN"))
                    for j in range(fill.shape[1])
                ]
            ).transpose(0, 1)

        return idx, age, logits
