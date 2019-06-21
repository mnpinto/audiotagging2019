from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *

__all__= ['XResNetssa', 'xresnet18ssa', 'xresnet34ssa', 'xresnet50ssa', 'xresnet101ssa', 'xresnet152ssa']

# XResnet with Simple Self Attention taken from: https://github.com/sdoria/SimpleSelfAttention

def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

act_fn = nn.ReLU(inplace=True)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1):#, n_out:int):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))
        
    def forward(self,x):
        size = x.size()
        x = x.view(*size[:2],-1)
        o = torch.bmm(x.permute(0,2,1).contiguous(),self.conv(x))
        o = self.gamma * torch.bmm(x,o) + x
        return o.view(*size).contiguous()

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1,sa=False):
        super().__init__()
        
        
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
                
        ]
        
        self.sa = SimpleSelfAttention(nf,ks=1) if sa else noop
        
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): 
        
        
        return act_fn(self.sa(self.convs(x)) + self.idconv(self.pool(x)))
    
class XResNetssa(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(nfs[i], nfs[i+1], stride=2 if i==0 else 1)
            for i in range(3)]

        nfs = [64//expansion,64,128,256,512]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1],
                                      n_blocks=l, stride=1 if i==0 else 2, sa = True if i in[len(layers)-4] else False)
                  for i,l in enumerate(layers)]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, sa = False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1, sa if i in [n_blocks -1] else False)
              for i in range(n_blocks)])
    
def xresnet18ssa(**kwargs): return XResNetssa.create(1, [2, 2, 2, 2], **kwargs)       
def xresnet34ssa(**kwargs): return XResNetssa.create(1, [3, 4, 6, 3], **kwargs)   
def xresnet50ssa(**kwargs): return XResNetssa.create(4, [3, 4, 6, 3], **kwargs)
def xresnet101ssa(**kwargs): return XResNetssa.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152ssa(**kwargs): return XResNetssa.create(4, [3, 8, 36, 3], **kwargs)