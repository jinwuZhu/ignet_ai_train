import torch.optim as optim
import ignet_ai_train.optim.modules as ignet_optim

_registered_optimizers = {}
def _init_register_optimizer():
    global _registered_optimizers
    from_packages = [optim,ignet_optim]
    for from_ in from_packages:
        for name in dir(from_):
            if name.startswith("_"):
                continue
            obj = getattr(from_, name)
            if isinstance(obj, type):
                _registered_optimizers[name] = obj
    # ... more
def create_optimizer(name,*args,**kwargs):
    if not _registered_optimizers: _init_register_optimizer()
    if name in _registered_optimizers:
        return _registered_optimizers[name](*args,**kwargs)
    else:
        raise Exception(f"Unknown optimizer: {name}")
