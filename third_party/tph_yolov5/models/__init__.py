# torch>=1.12 compatibility: the released TPH-YOLOv5 checkpoints pickle nn.GELU
# modules created with torch 1.x, which lack the 'approximate' attribute.
import torch
import torch.nn as nn


def _gelu_forward(self, input):
    return torch.nn.functional.gelu(input, approximate=getattr(self, 'approximate', 'none'))


nn.GELU.forward = _gelu_forward
