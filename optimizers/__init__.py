from .adam import Adam, Adam4bit, Adam8bit, AdamFp8
from .adamw import AdamW, AdamW4bit, AdamW8bit, AdamWFp8


class OptimizerInBackward:
    def __init__(self, params, optim_cls, **kwargs):
        params = list(params)
        self.optim_dict = dict()

        def backward_hook(p):
            self.optim_dict[p].step()
            self.optim_dict[p].zero_grad()

        for p in params:
            self.optim_dict[p] = optim_cls([p], **kwargs)
            p.register_post_accumulate_grad_hook(backward_hook)

    def step(self, closure=None):
        return closure() if closure is not None else None

    def zero_grad(self, set_to_none=True):
        return

    @property
    def param_groups(self):
        return sum((optim.param_groups for optim in self.optim_dict.values()), start=[])
