import os
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn.functional as F
from delphi import DAYS_PER_YEAR
from delphi.config import dataclass_from_dict
from model import DelphiConfig, Delphi
import numpy as np
from utils import get_batch, get_p2i

@dataclass
class GeneratorConfig: 
    device: str = 'cpu'
    batch_size: int = 64
    no_repeat: bool = True
    top_k: int = 0
    temperature: float = 1.0
    max_age_in_years: int = 80
    max_new_tokens: int = 100
    simulate_comorbid: bool = True
    comorbid_cutoff: float = 0.2
    
@dataclass
class PromptConfig:
    data_memmap: str = 'data/ukb_simulated_data/val.bin'
    start_age_in_years: int = 60

def validate_generator_config():
    pass

class Generator:
    
    def __init__(
        self, 
        cfg: GeneratorConfig, 
        model: Delphi
        ):
        self.config = cfg
        self.model = model
    
    @torch.no_grad()
    def generate_one_batch(
        self, 
        idx: torch.Tensor, 
        age: torch.Tensor, 
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if self.config.max_new_tokens == -1:
            self.config.max_new_tokens = 10000
        for _ in range(self.config.max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx #if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            age_cond = age #if age.size(1) <= self.config.block_size else age[:, -self.config.block_size:]

            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self.model(idx_cond, age_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.config.temperature
            logits[:,self.model.config.ignore_tokens] = -float('Inf')
            # optionally crop the logits to only the top k options
            if self.config.top_k > 0:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if self.config.no_repeat: # TODO: find out why no_repeat would be set to true
                fill = idx + 0
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -float("Inf"))
            
            probs = F.softmax(logits, dim=-1)
            comorbid_count = (probs > self.config.comorbid_cutoff).sum(-1)
            max_comorbid = int(comorbid_count.max().item())
            if self.config.simulate_comorbid and max_comorbid > 1:
                idx_next = torch.zeros(
                    idx_cond.shape[0], 
                    max_comorbid, 
                    device=idx.device, dtype=torch.long)
                age_next = torch.zeros_like(idx_next, dtype=torch.float32)
                for n_comorbid in range(1, max_comorbid+1):
                    subject_idx, comorbid_tokens = torch.nonzero(
                        torch.logical_and(probs > self.config.comorbid_cutoff, (comorbid_count == n_comorbid).unsqueeze(-1)),
                        as_tuple=True)
                    subject_idx = torch.unique(subject_idx)
                    assert comorbid_tokens.size().numel() % n_comorbid == 0
                    comorbid_tokens = comorbid_tokens.view(-1, n_comorbid)
                    idx_next[subject_idx, :n_comorbid] = comorbid_tokens 
                    comorbid_logits = torch.gather(logits, dim=1, index=comorbid_tokens)
                    t_next = torch.clamp(
                        -torch.exp(-comorbid_logits) * torch.rand(comorbid_logits.shape, device=idx.device).log(),
                        min=0, max=DAYS_PER_YEAR*80.
                        )
                    t_next = t_next[:, torch.randperm(n_comorbid)]
                    age_next[subject_idx, :n_comorbid] = age_cond[subject_idx, -1].unsqueeze(-1) + t_next 
            else: 
                t_next = torch.clamp(
                    -torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(),
                    min=0, max=DAYS_PER_YEAR*80.
                ).min(1)
                #age_next = age[...,[-1]] + torch.clamp(-torch.exp(-lse) * torch.rand(lse.shape, device=idx.device).log(), min=self.config.t_min, max=365*80.) #torch.normal(torch.zeros((1,1), device=idx.device),1.)
                idx_next = t_next[1][:,None]
                age_next = age[...,[-1]] + t_next[0][:,None]
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # todo: should we check for if death token has been generated instead of if idx_next is death
            if torch.all(idx_next == self.model.config.vocab_size -1) or torch.all(age_next > self.config.max_age_in_years*DAYS_PER_YEAR):
                break
        
        pad = (torch.cumsum(
            torch.cumsum(idx == self.model.config.vocab_size -1, dim=1).int(),
            dim=1
            ) > 1) + (age > self.config.max_age_in_years*DAYS_PER_YEAR)
        logits, _, _ = self.model(idx, age)
        idx[pad] = 0
        age[pad] = float('NaN')
        if self.config.no_repeat: # TODO: find out if this is repetitive
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([logits[:,j].scatter_(1, fill[:,:j+1], float("NaN")) for j in range(fill.shape[1])]).transpose(0,1)

        return idx, age, logits
    
    def generate(
        self, 
        d0: torch.Tensor,
        d1: torch.Tensor,
        model: Delphi, 
        ) -> tuple[np.ndarray, np.ndarray]: 
        
        seed = 1337
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        n_repeats = 1
        oo = []
        model.to(self.config.device)
        for _ in range(n_repeats):
            with torch.no_grad():
                for dd in tqdm(zip(*map(lambda x: torch.split(x, self.config.batch_size), (d0,d1))), total=len(d0)//self.config.batch_size + 1):
                    mm = self.generate_one_batch(
                        dd[0].to(self.config.device), # TODO: check if moving to device is redundant here
                        dd[1].to(self.config.device), 
                        )
                    oo += [(mm[0],mm[1])]

        max_len = max(x[1].shape[1] for x in oo)

        a = [np.pad(x[0].cpu(), ((0,0), (0, max_len - x[0].shape[1])), constant_values=0) for x in oo]
        b = [np.pad(x[1].cpu(), ((0,0), (0, max_len - x[1].shape[1])), constant_values=-10000) for x in oo]

        # Concatenate along first dimension
        a = np.concatenate(a, axis=0)
        print(a.shape)
        b = np.concatenate(b, axis=0) / DAYS_PER_YEAR
        b = np.nan_to_num(b, nan=-27).astype('int')
            
        return a, b
    
def load_model(
    ckpt_path,
    model_cls=Delphi,
    model_cfg_cls=DelphiConfig,
    ):
    
    ckpt_path = Path(ckpt_path)
    train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
    ckpt_dict = torch.load(
        ckpt_path / "ckpt.pt", 
        map_location=torch.device('cpu') if not torch.cuda.is_available() else None
    )

    param_dtype = dict(
        float32=torch.float32,
        float64=torch.float64, 
        float16=torch.float16, 
        bfloat16=torch.bfloat16)[
        train_cfg.dtype
    ]
    model_cfg = dataclass_from_dict(model_cfg_cls, train_cfg.model, strict=False)
    model = model_cls(model_cfg)
    model.load_state_dict(ckpt_dict["model"])
    model = model.eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    
    return model, train_cfg

def load_prompt(
    data_memmap: str, 
    start_age_in_years: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    
    val = np.fromfile(data_memmap, dtype=np.uint32).reshape(-1,3)

    val_p2i = get_p2i(val)
    
    d = get_batch(
        ix=range(0,val_p2i.shape[0]-1,1), 
        data=val, 
        p2i=val_p2i,  
        select='left', 
        block_size=63, 
        device='cpu', # TODO: fix this with a proper device
        padding='random')
    
    n_samples = 1024 * 1024 #TODO: find out the purpose of this
    
    start_age_in_days = start_age_in_years * DAYS_PER_YEAR

    w = np.where(
        (d[1].cpu().detach().numpy() <= start_age_in_days).any(1) * (d[3].cpu().detach().numpy() >= start_age_in_days).any(1)
        )
    # select everything
    # w = np.arange(d[0].shape[0])[None]
    u = np.unique(w[0])

    d0 = d[0][u[:n_samples]].clone().detach()
    d1 = d[1][u[:n_samples]].clone().detach()

    d0[d1>start_age_in_days] = 0
    d1[d1>start_age_in_days] = -10000.

    if start_age_in_years > 0:
        d0 = torch.nn.functional.pad(d0, (0,1), 'constant', 1)
        d1 = torch.nn.functional.pad(d1, (0,1), 'constant', start_age_in_days)

    o = d1.argsort(1)
    d0 = d0.gather(1, o)
    d1 = d1.gather(1, o)
    
    return d0, d1

def main(): 
    
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        GeneratorConfig, cfg, strict=False
    )
    print(gen_cfg)
    prompt_cfg = dataclass_from_dict(
        PromptConfig, cfg, strict=False
    )
    print(prompt_cfg)
    
    os.makedirs(cfg.dump_dir, exist_ok=True)
    with open(os.path.join(cfg.dump_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    
    model, _ = load_model(cfg.ckpt_path)
    model = model.to(gen_cfg.device)
    
    gen = Generator(cfg=gen_cfg, model=model)
    
    d0, d1 = load_prompt(
        data_memmap=prompt_cfg.data_memmap, 
        start_age_in_years=prompt_cfg.start_age_in_years
        )
    
    a, b = gen.generate(model=model, d0=d0, d1=d1)
    np.save(arr=a, file=os.path.join(cfg.dump_dir, 'token.npy'))
    np.save(arr=b, file=os.path.join(cfg.dump_dir, 'time.npy'))

if __name__ == "__main__": 
    main()