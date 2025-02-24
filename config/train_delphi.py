
import time

out_dir = 'Delphi'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 25
log_interval = 25 # don't print too too often
seed = 42

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'delphi'
wandb_run_name = 'run' + str(time.time())

dataset = 'ukb_real_data'
batch_size = 128
block_size = 48
data_fraction = 1.0

n_layer = 12
n_head = 12
n_embd = 120
dropout = 0.1
weight_decay = 2e-1
vocab_size = 1270

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially
ignore_tokens = [0, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # padding, sex and lifestyle tokens
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
