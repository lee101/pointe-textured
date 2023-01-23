import torch


def convert_float_to_bfloat16(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.float()
        if isinstance(layer, torch.nn.Linear):
            layer.float()
    # check if fields are other than float32, only convert float32 to bfloat16
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
        if param.dtype == torch.float:
            param.data = param.data.to(torch.bfloat16)
    return model
