import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class SpectralNorm:
    def __init__(self, name, bound=False):
        self.name = name
        self.bound = bound

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)

        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()

        sigma = u @ weight_mat @ v

        if self.bound:
            weight_sn = weight / (sigma + 1e-6) * torch.clamp(sigma, max=1)

        else:
            weight_sn = weight / sigma

        return weight_sn, u

    @staticmethod
    def apply(module, name, bound):
        fn = SpectralNorm(name, bound)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, init=True, std=1, bound=False):
    if init:
        nn.init.normal_(module.weight, 0, std)

    if hasattr(module, 'bias') and module.bias is not None:
        module.bias.data.zero_()

    SpectralNorm.apply(module, 'weight', bound=bound)

    return module


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_class=None, downsample=False):
        super().__init__()

        self.conv1 = weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            ), 'weight'
        )

        self.conv2 = weight_norm(
            nn.Conv2d(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=False if n_class is not None else True,
            ),'weight'
        )

        self.class_embed = None

        if n_class is not None:
            class_embed = nn.Embedding(n_class, out_channel * 2 * 2)
            class_embed.weight.data[:, : out_channel * 2] = 1
            class_embed.weight.data[:, out_channel * 2 :] = 0

            self.class_embed = class_embed

        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(
                weight_norm(nn.Conv2d(in_channel, out_channel, 1, bias=False),'weight')
            )

        self.downsample = downsample

    def forward(self, input, class_id=None):
        out = input

        out = self.conv1(out)

        if self.class_embed is not None:
            embed = self.class_embed(class_id).view(input.shape[0], -1, 1, 1)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1

        out = F.silu(out)
        out = self.conv2(out)

        if self.class_embed is not None:
            out = weight2 * out + bias2

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = F.silu(out)

        return out


class IGEBM(nn.Module):
    def __init__(self, n_class=None, dataset='mnist', in_channels=1):
        super().__init__()
        self.dataset = dataset
        self.in_channels = in_channels
        if self.dataset == 'celeba':
            
            self.conv1 = weight_norm(nn.Conv2d(3, 64, 3, padding=1),'weight')
            self.blocks = nn.ModuleList(
                [
                    ResBlock(64, 64, n_class, downsample=True),
                    ResBlock(64, 64, n_class),
                    ResBlock(128, 128, n_class, downsample=True),
                    ResBlock(128, 128, n_class),
                    ResBlock(128, 128, n_class, downsample=True),
                    ResBlock(128, 256, n_class),
                    ResBlock(256, 256, n_class, downsample=True),
                    ResBlock(256, 256, n_class),
                ]
            )
            self.linear = nn.Linear(256, 1)
    
        else:
            
            self.conv1 = weight_norm(nn.Conv2d(self.in_channels, 128, 3, padding=1),'weight')
            self.blocks = nn.ModuleList(
                [
                    ResBlock(128, 128, n_class, downsample=True),
                    ResBlock(128, 128, n_class),
                    ResBlock(128, 256, n_class, downsample=True),
                    ResBlock(256, 256, n_class),
                    ResBlock(256, 256, n_class, downsample=True),
                    ResBlock(256, 256, n_class),
                ]
            )
            self.linear = nn.Linear(256, 1)

            self.all_conv_layers = []
            for n, layer in self.named_modules():
                if isinstance(layer, Conv2D):
                    self.all_conv_layers.append(layer)

    def forward(self, input, class_id=None):
        out = self.conv1(input)
        out = F.silu(out)

        for block in self.blocks:
            out = block(out, class_id)

        out = F.silu(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
    
    def norm_loss(self):
        loss = 0
        for l in self.all_conv_layers:
            loss += l.get_spectral_norm()
        return loss
