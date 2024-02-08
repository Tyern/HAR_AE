from torch import nn

def unwrap_model(model):
    layer = []
    def _unwrap_model(model):
        # nonlocal layer
        for i in model.children():
            if isinstance(i, nn.Sequential): _unwrap_model(i)
            else: layer.append(i)
    _unwrap_model(model)
    return layer