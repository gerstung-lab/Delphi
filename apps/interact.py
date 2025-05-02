from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf

from delphi import DAYS_PER_YEAR
from delphi.model.transformer import load_model
from delphi.sampler import CausalSampler, CausalSamplerConfig
from delphi.tokenizer import CoreEvents, load_tokenizer_from_ckpt


def parse_prompt(prompt: str):

    prompt = prompt.strip()
    prompt = prompt.replace(" ", "")

    age_token_lst = prompt.split(";")
    events = [age_token.split(",")[0] for age_token in age_token_lst]
    timesteps = [float(age_token.split(",")[1]) for age_token in age_token_lst]

    return events, timesteps


def pack(tokens: list[np.ndarray], timesteps: list[np.ndarray]):

    max_len = max([len(t) for t in tokens])
    for i in range(len(tokens)):
        tokens[i] = np.pad(
            tokens[i],
            (0, max_len - len(tokens[i])),
            mode="constant",
            constant_values=0,
        )
        timesteps[i] = np.pad(
            timesteps[i],
            (0, max_len - len(timesteps[i])),
            mode="constant",
            constant_values=-10000,
        )

    X = np.stack(tokens, axis=0)
    T = np.stack(timesteps, axis=0)

    return X, T


def unpack(X: np.ndarray, T: np.ndarray):

    tokens = []
    timesteps = []
    for i in range(X.shape[0]):
        tokens.append(X[i][X[i] != 0])
        timesteps.append(T[i][T[i] != -10000])

    return tokens, timesteps


def main():

    cli_args = OmegaConf.from_cli()
    ckpt = cli_args.ckpt

    tokenizer = load_tokenizer_from_ckpt(ckpt)

    model, _ = load_model(ckpt)
    model.eval()
    model.to("cpu")

    sampler_cfg = CausalSamplerConfig(termination_tokens=[CoreEvents.DEATH.value])
    sampler = CausalSampler(
        model=model,
        tokenizer=tokenizer,
        cfg=sampler_cfg,
    )

    # prompts = []
    # while True:
    #     prompt = input("Enter a prompt (or press enter to finish): ")
    #     if not prompt:
    #         break
    #     prompts.append(prompt)
    prompts = ["male, 0; a38_(scarlet_fever), 10"]
    print(prompts)

    X_lst = []
    T_lst = []
    for prompt in prompts:
        events, timesteps_batch = parse_prompt(prompt)
        tokens_batch = tokenizer.encode(events)
        timesteps_batch = np.array(timesteps_batch, dtype=np.float32) * DAYS_PER_YEAR
        X_lst.append(tokens_batch)
        T_lst.append(timesteps_batch)

    X, T = pack(X_lst, T_lst)

    tokens_batch, timesteps_batch, _ = sampler.generate(
        age=torch.Tensor(T),
        idx=torch.Tensor(X).to(torch.long),
    )

    tokens_batch, timesteps_batch = unpack(
        X=tokens_batch.cpu().numpy().astype(np.uint32),
        T=timesteps_batch.cpu().numpy(),
    )

    for tokens, timesteps in zip(tokens_batch, timesteps_batch):

        tokens = tokens.tolist()
        events = tokenizer.decode(tokens)
        timesteps /= DAYS_PER_YEAR
        timesteps = timesteps.tolist()

        for event, time in zip(events, timesteps):
            print(f"{event}, {time:.2f}")
    # tokens, timesteps, _ = sampler.generate()


if __name__ == "__main__":
    main()
