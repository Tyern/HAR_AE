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

def reset_weight_model(model, verbose=0):
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            reset_weight_model(layer, verbose = verbose)
        else:
            if hasattr(layer, 'reset_parameters'):
                if verbose: print("reset", type(layer))
                layer.reset_parameters()
