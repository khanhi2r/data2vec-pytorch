import torch
import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMA.
    """
    def __init__(self, model: nn.Module, cfg, device=None, skip_keys=None):
        self.model = model
        self.model.requires_grad_(False)
        self.model.to(device)
        self.cfg = cfg
        self.device = device
        self.skip_keys = skip_keys or set()
        self.decay = self.cfg.ema_decay
        self.num_updates = 0

    def step(self, new_model: nn.Module):
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key]
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model