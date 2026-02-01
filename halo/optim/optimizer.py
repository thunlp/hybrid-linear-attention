from torch import nn
import torch


def get_optimizer(
    model: nn.Module,
    weight_decay: float,
    lr: float,
    beta1: float,
    beta2: float,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    '''
    Create an AdamW optimizer.
    '''
    # start with all candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # This will use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=(beta1, beta2),
    )

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if verbose:
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
    return optimizer
