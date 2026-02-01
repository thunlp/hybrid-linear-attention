from torch import nn


def get_num_params(model: nn.Module, non_embedding: bool = False) -> int:
    """
    Get the number of parameters in the model.
    """
    cnt = 0
    for n, p in model.named_parameters():
        if non_embedding:
            if "embed" in n or "emb." in n:
                continue
        cnt += p.numel()
    return cnt


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    Accounts for DeepSpeed's ZeRO-3 partitioned parameters.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel  # Get the true number of elements for ZeRO-3 parameters
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if all_param > 0:
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")
