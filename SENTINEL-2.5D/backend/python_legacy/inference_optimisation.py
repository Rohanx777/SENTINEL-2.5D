"""Inference-only algebraic BatchNorm folding, including activation-before-BN blocks.
Keep checkpoint loading strict before this transform. No clipping or fabricated weights.
"""
import torch
from torch import nn

class AffineAfterActivation(nn.Module):
    def __init__(self,sign,shift):
        super().__init__();self.register_buffer('sign',sign);self.register_buffer('shift',shift)
    def forward(self,value): return value*self.sign+ self.shift

def fold(layer,bn,activation_before=False):
    scale=bn.weight.detach()/torch.sqrt(bn.running_var+bn.eps)
    shift=bn.bias.detach()-bn.running_mean*scale
    weight_scale=scale.abs() if activation_before else scale
    shape=[len(scale)]+[1]*(layer.weight.ndim-1)
    layer.weight=nn.Parameter(layer.weight.detach()*weight_scale.reshape(shape),requires_grad=False)
    old_bias=layer.bias.detach() if layer.bias is not None else torch.zeros_like(scale)
    bias=old_bias*weight_scale
    if not activation_before: bias=bias+shift
    layer.bias=nn.Parameter(bias,requires_grad=False)
    return AffineAfterActivation(torch.sign(scale),shift) if activation_before else nn.Identity()

def optimise_model(model,half=False):
    pp=model.cylinder_3d_generator.PPmodel
    for linear,batchnorm in [(1,2),(4,5),(7,8)]: pp[batchnorm]=fold(pp[linear],pp[batchnorm])
    for parent in model.cylinder_3d_spconv_seg.modules():
        name=type(parent).__name__
        if name in ('ResContextBlock','ResBlock'):
            pairs=[('conv1','bn0'),('conv2','bn1'),('conv3','bn2')]
            if hasattr(parent,'conv1_2'): pairs.append(('conv1_2','bn0_2'))
            before=True
        elif name=='UpBlock':
            pairs=[('trans_dilao','trans_bn'),('conv1','bn1'),('conv2','bn2'),('conv3','bn3')];before=True
        elif name=='ReconBlock':
            pairs=[('conv1','bn0'),('conv1_2','bn0_2'),('conv1_3','bn0_3')];before=False
        else: continue
        for conv,bn in pairs: setattr(parent,bn,fold(getattr(parent,conv),getattr(parent,bn),before))
    if half:
        # The released model's latent point features exceed FP16 range.
        # Positive homogeneous rescaling is cancelled in BOTH first-conv branches.
        compression=model.cylinder_3d_generator.fea_compression[0]
        compression.weight.data.div_(256)
        compression.bias.data.div_(256)
        for name in ('conv1','conv2'):
            getattr(model.cylinder_3d_spconv_seg.downCntx,name).weight.data.mul_(256)
        model.half()
        # Input normalization statistics retain FP32 range.
        for module in model.modules():
            if isinstance(module,nn.BatchNorm1d): module.float()
    return model.eval()
