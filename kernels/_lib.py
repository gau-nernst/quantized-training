import torch

lib = torch.library.Library("qtrain", "DEF")
lib_ops = torch.ops.qtrain
