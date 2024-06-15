import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import Delphi, DelphiConfig


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'delphi'
wandb_run_name = 'run' + str(time.time())
# data
dataset = 'ukb_data'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24
# model
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32' #'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

token_dropout = 0.0
t_min = 0. #365.25/12.
mask_ties = True
ignore_tokens = [0]
data_fraction = 1.0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE']) # total number of training processes
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    world_size = 1
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1339)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32,'float64': torch.float64, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1,3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1,3)

def get_p2i(data):
    px = data[:,0].astype('int')
    pix = sorted(list(set(px)))
    #p2i = np.array([(p, (px==p).argmax(), (px==p).sum()) for i,p in enumerate(pix)])
    p2i = []
    j = 0
    q = px[0]
    for i,p in enumerate(px):
        if p != q:
            p2i.append([j,i-j])
            q = p
            j = i
    return np.array(p2i)

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    p2i = train_p2i if split == 'train' else val_p2i
    mask_time = -10000.
    #ix = torch.randint(data.shape[0] - block_size-1, (batch_size,))
    if split == 'train':
        ix = torch.randint(int(len(p2i) * data_fraction), (batch_size,))
    else:
        ix = torch.randint(len(p2i), (batch_size,))
        
    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    this_i = block_size//2

    ix = torch.clamp(x[:,0] + (ix.sort().values % x[:,1]) - this_i, 0, data.shape[0] - block_size - 1)
    
    mask = torch.from_numpy(np.stack([data[i:i+block_size+1,0] == data[i+this_i,0] for i in ix]))
    
    tokens = torch.stack([torch.from_numpy(data[i:i+block_size+1,2].astype(np.int64)) for i in ix]) # Mask 0 when patients change
    ages = torch.stack([torch.from_numpy(data[i:i+block_size+1,1].astype(np.float32)) for i in ix])
    
    tokens = tokens * mask - (1-1*mask)
    ages = ages * mask + mask_time * (1-1*mask)
    
    pad = torch.randint(36525, (len(ix), 20))
    
    m = ages.max(1,keepdim=True).values
    
    tokens = torch.hstack([tokens,torch.zeros(len(ix), pad.shape[1], dtype=torch.int)])
    ages = torch.hstack([ages, pad])
    
    lifestyle_idx = (tokens >= 3) * (tokens <= 11)

    if split == 'train':
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (int(lifestyle_idx.sum()),))
        
    tokens = tokens.masked_fill(ages>m, -1)
    ages = ages.masked_fill(ages>m, mask_time)
    
    s = torch.argsort(ages + (tokens!=-1) + (tokens==model.config.vocab_size-2) + torch.rand(ages.shape), 1)    
    tokens = torch.gather(tokens,1,s) + 1
    ages = torch.gather(ages,1,s)
           
    i = pad.shape[1]
    x = tokens[:,i:-1]
    a = ages[:,i:-1] #/ 365.25
    y = tokens[:,i+1:]
    b = ages[:,i+1:] #/ 365.25
    
    x = x.masked_fill(torch.logical_and(x==0, y==1), 1)    
    y = y.masked_fill(x==0, 0)
    b = b.masked_fill(x==0, mask_time)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = x.pin_memory().to(device, non_blocking=True), a.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), b.pin_memory().to(device, non_blocking=True)
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, token_dropout=token_dropout, t_min=t_min, mask_ties=mask_ties, ignore_tokens=ignore_tokens) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
    
if block_size != model.config.block_size:
    model.adjust_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.config.block_size = block_size
    
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, A, Y, B = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, A, Y, B)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, A, Y, B = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

val_loss = None
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss()
        if val_loss is None:
            val_loss = losses['val']
        val_loss = 0.5 * losses['val'] + 0.5 * val_loss # ie exponential decay
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ({val_loss:.4f})")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/agg_loss": losses['train'],
                "val/loss": losses['val'],
            })
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, att = model(X, A, Y, B)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, A, Y, B = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * world_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": loss,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "weights": wandb.Histogram(model.transformer.wte.weight.cpu().detach().numpy()),
                "logits": wandb.Histogram(logits.cpu().detach().numpy()),
            })
        
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
