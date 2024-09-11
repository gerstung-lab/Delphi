import numpy as np
import torch
import re


def get_p2i(data):
    """
    Get the patient to index mapping.
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
    return np.array(p2i)


def pad_to_length(a, length):
    if len(a) >= length:
        return a
    return np.pad(a, (0, length - len(a)), 'constant', constant_values=-1)


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'center', 'right', 'smart_random', 'smart_right'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
    """

    mask_time = -10000.

    if select == 'center':
        this_i = block_size // 2
    elif select == 'right':
        this_i = block_size
    elif select in ['smart_random', 'smart_right']:
        this_i = 0
    else:
        raise NotImplementedError

    if index == 'patient':
        x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
        ix = torch.tensor(np.array(ix))

        gen = torch.Generator(device='cpu')
        # we want some things be random, but also deterministic
        gen.manual_seed(ix.sum().item())

        if select in ['center', 'right']:
            ix = torch.clamp(x[:, 0] + (torch.randint(0, 100, ix.shape,
                             generator=gen) % x[:, 1]) - this_i, 0, data.shape[0])
        else:
            if select == 'smart_random':
                ix = torch.clamp(x[:, 0] + (torch.randint(0, 10000, ix.shape, generator=gen) % torch.maximum(
                    torch.ones([1], dtype=torch.int), x[:, 1] - block_size)) - this_i, 0, data.shape[0])
            else:
                ix = torch.clamp(x[:, 0] + torch.maximum(torch.ones(
                    [1], dtype=torch.int), x[:, 1] - block_size) - this_i, 0, data.shape[0])
    else:
        raise NotImplementedError

    # print([(data[i:i + block_size + 1, 0] == data[i + this_i, 0]).shape for i in ix])
    mask = torch.from_numpy(np.stack([pad_to_length(data[i:i + block_size + 1, 0], block_size+1) == data[i + this_i, 0] for i in ix]))

    tokens = torch.stack([torch.from_numpy(pad_to_length(data[i:i + block_size + 1, 2], block_size+1).astype(np.int64)) for i in ix])
    ages = torch.stack([torch.from_numpy(pad_to_length(data[i:i + block_size + 1, 1], block_size+1).astype(np.float32)) for i in ix])

    # augment lifestyle tokens to avoid immortality bias
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (int(lifestyle_idx.sum()),))

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # insert a "no event" token every 5 years on average
    if padding == 'regular':
        pad = torch.arange(0, 36525, 3652.5 / 2) * torch.ones(len(ix), 1)
    elif padding in 'random':
        pad = torch.randint(36525, (len(ix), 20), generator=gen) + 1
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros(len(ix), pad.shape[1], dtype=torch.int)])
    ages = torch.hstack([ages, pad])
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    mask2 = (ages == 0) * (tokens < 1)
    ages = ages.masked_fill(mask2, mask_time)
    tokens = tokens.masked_fill(mask2, -1)

    s = torch.argsort(ages + (tokens != -1) + (tokens == block_size - 2) + torch.rand(ages.shape, generator=gen), 1)
    tokens = torch.gather(tokens, 1, s) + 1
    ages = torch.gather(ages, 1, s)

    # shift by one to generate targets
    i = pad.shape[1]
    x = tokens[:, i:-1]
    a = ages[:, i:-1]  # / 365.25
    y = tokens[:, i + 1:]
    b = ages[:, i + 1:]  # / 365.25

    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
    return x, a, y, b


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f
