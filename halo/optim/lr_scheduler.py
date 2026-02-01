import math
import torch


def get_wsd_lr(
    it: int,
    lr: float,
    min_lr: float,
    n_drop_iters: int,
    n_warmup_iters: int,
    n_train_iters: int,
) -> float:
    '''
    WSD scheduler, linear warmup and cosine decay.
    '''
    # 1) Warmup stage
    if it < n_warmup_iters:
        return lr * it / n_warmup_iters
    # 2) Stable stage: if it < n_train_iters - n_decay_iters: return max_lr
    if it < n_train_iters - n_drop_iters:
        return lr
    # 3) Cosine Annealing stage
    if it < n_train_iters:
        decayed_steps = it - n_train_iters + n_drop_iters
        decay_ratio = decayed_steps / n_drop_iters
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (lr - min_lr)
    # 4) After annealing
    return min_lr


# learning rate decay scheduler (cosine with warmup)
def get_cos_lr(
    it: int, lr: float, min_lr: float, n_warmup_iters: int, n_train_iters: int
) -> float:
    return get_wsd_lr(
        it,
        lr=lr,
        min_lr=min_lr,
        n_drop_iters=n_train_iters - n_warmup_iters,
        n_warmup_iters=n_warmup_iters,
        n_train_iters=n_train_iters,
    )


def get_constant_lr(it: int, lr: float) -> float:
    return lr


class WSDScheduler(torch.optim.lr_scheduler.LRScheduler):
    '''
    The Warmup-Stable-Decay (WSD) learning rate scheduler.
    Linear warmup and cosine decay.
    '''
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        n_decay_iters: int,
        n_warmup_iters: int,
        n_train_iters: int,
        min_lr: float | None = None,
    ):
        '''
        min_lr: The LR at the end of the decay phase. Default is lr/10.
        '''
        assert isinstance(n_warmup_iters, int)
        assert isinstance(n_decay_iters, int)
        assert isinstance(n_train_iters, int)
        assert n_warmup_iters >= 0
        assert n_decay_iters >= 0
        assert n_train_iters >= 0
        assert n_warmup_iters <= n_train_iters - n_decay_iters, 'n_warmup_iters must be <= n_train_iters - n_decay_iters'
        self.max_lr = lr
        if min_lr is None:
            self.min_lr = lr / 10
        else:
            self.min_lr = min_lr
        self.n_warmup_iters = n_warmup_iters
        self.n_decay_iters = n_decay_iters
        self.n_train_iters = n_train_iters
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> list[float]:
        """
        learning rate decay scheduler (cosine with warmup)
        """
        if self._step_count < self.n_warmup_iters:
            # 1) Linear warmup for warmup_iters steps
            lr = self.max_lr * self._step_count / self.n_warmup_iters
        elif self._step_count < self.n_train_iters - self.n_decay_iters:
            # 2) Stable phase
            lr = self.max_lr
        elif self._step_count < self.n_train_iters:
            # 3) Cosine decay
            decayed_steps = self._step_count - (self.n_train_iters - self.n_decay_iters)
            decay_ratio = decayed_steps / self.n_decay_iters
            # assert 0 <= decay_ratio <= 1
            assert 0 <= decay_ratio
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        else:
            # 4) After annealing
            lr = self.min_lr
        # return list of learning rates to be applied on every parameter group
        return [lr] * len(self.optimizer.param_groups)
