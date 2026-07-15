from abc import abstractmethod
from functools import partial
import math

import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer, GatedSelfAttentionDense, GatedSelfAttentionDense2
# from .positionnet  import PositionNet
from torch.utils import checkpoint
from ldm.util import instantiate_from_config
from copy import deepcopy

from einops import rearrange
from torch import Tensor
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, objs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, objs)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x




class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, emb , use_reentrant=False )
        else:
            return self._forward(x, emb)


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


############################MASK####################################
class DecoderWeightsMaskGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, factor=16, temb_dim=1280,
                 sem_in_dim=2, use_sem_cond=True):
        super().__init__()
        assert factor >= 1
        assert in_channels % factor == 0, f"in_channels={in_channels} not divisible by factor={factor}"
        assert out_channels % factor == 0, f"out_channels={out_channels} not divisible by factor={factor}"

        new_out = out_channels // factor
        new_in = in_channels // factor

        self.conv_kernel_mask = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, new_in * new_out, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.proj_temb = nn.Linear(temb_dim, in_channels, bias=False)

        self.use_sem_cond = use_sem_cond
        self.sem_in_dim = sem_in_dim
        if self.use_sem_cond:
            self.proj_sem = nn.Conv2d(sem_in_dim, in_channels, kernel_size=1, bias=False)
            nn.init.constant_(self.proj_sem.weight, 0.0)

        self.in_c = new_in
        self.out_c = new_out
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            # 最后一层输出 mask 的 1x1 conv：改为小随机初始化
            if m.kernel_size == (1, 1) and m.out_channels == self.in_c * self.out_c:
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            else:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='linear')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, weight_shape, sample, res_sample, temb, encoder_hidden_states,
                grounding_extra_input=None, layer_name=None):
        flag = len(weight_shape)
        temb = self.proj_temb(temb).unsqueeze(-1).unsqueeze(-1)

        if res_sample is None:
            x = sample
        else:
            x = th.cat([sample, res_sample], dim=1)

        if flag == 4:
            _, _, k_h, k_w = weight_shape
            x = F.adaptive_avg_pool2d(x, (k_h, k_w))
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        sem_cond = 0.0
        if self.use_sem_cond and (grounding_extra_input is not None):
            sem = grounding_extra_input
            if sem.dim() != 4:
                raise ValueError(f"grounding_extra_input must be 4D [B,C,H,W], got {tuple(sem.shape)}")
            if sem.shape[1] != self.sem_in_dim:
                raise ValueError(f"grounding_extra_input channel mismatch: expect {self.sem_in_dim}, got {sem.shape[1]}")

            if flag == 4:
                sem = F.adaptive_avg_pool2d(sem, (k_h, k_w))
            else:
                sem = F.adaptive_avg_pool2d(sem, (1, 1))

            sem_cond = self.proj_sem(sem)

        x = x + temb + sem_cond
        mask = self.conv_kernel_mask(x)

        if flag == 4:
            mask = mask.view(mask.size(0), self.out_c, self.in_c, k_h, k_w)
        else:
            mask = mask.view(mask.size(0), self.out_c, self.in_c)

        return mask


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = (
        -th.empty_like(logits, memory_format=th.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        y_hard = (y_soft > threshold).to(y_soft.dtype)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class Adapter(nn.Module):
    def __init__(
        self,
        out_c,
        in_c,
        new_out_c,
        new_in_c,
        tau=1.0,
        init_bias=0.0,         # 不再用 5.0
        init_weight_std=1e-3,  # 小随机初始化
    ):
        super(Adapter, self).__init__()
        self.conv_in = nn.Conv2d(out_c, new_out_c, kernel_size=1)
        self.conv_out = nn.Conv2d(in_c, new_in_c, kernel_size=1)
        self.tau = tau
        self.init_bias = init_bias
        self.init_weight_std = init_weight_std
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=self.init_weight_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, self.init_bias)

    def forward(self, input_tensor):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(-1).unsqueeze(-1)

        x = rearrange(input_tensor, 'b out_c in_c h w -> (b in_c) out_c h w')
        x = self.conv_in(x)
        x = rearrange(x, '(b in_c) new_out_c h w -> b new_out_c in_c h w', b=input_tensor.shape[0])
        x = rearrange(x, 'b new_out_c in_c h w -> (b new_out_c) in_c h w')
        x = self.conv_out(x)

        # 全程 hard
        x = gumbel_sigmoid(x, tau=self.tau, hard=True)

        x = rearrange(x, '(b new_out_c) new_in_c h w -> b new_out_c new_in_c h w', b=input_tensor.shape[0])

        if x.size(-1) == 1 and x.size(-2) == 1:
            x = x.squeeze(-1).squeeze(-1)
        return x


class AffineWeightCombiner(nn.Module):
    """
    Learnable affine combination for dynamic weight construction:
        W_hat = k0 * W + k1 * 1 + k2 * M
    where W is the pretrained/original weight tensor and M is the task-specific
    mask (after adapter). k0, k1, k2 are learnable scalars per hooked layer.
    """
    def __init__(self, init_k0=1.0, init_k1=0.0, init_k2=1e-2):
        super().__init__()
        self.k0 = nn.Parameter(th.tensor(float(init_k0)))
        self.k1 = nn.Parameter(th.tensor(float(init_k1)))
        self.k2 = nn.Parameter(th.tensor(float(init_k2)))

    def forward(self, weight, mask):
        if weight.dim() != mask.dim():
            raise ValueError(
                f"AffineWeightCombiner expects weight and mask with same dims, "
                f"got weight.dim={weight.dim()} mask.dim={mask.dim()}"
            )
        ones = th.ones_like(weight)
        return self.k0 * weight + self.k1 * ones + self.k2 * mask


def get_linear_hook_fn(layer_name=None):
    def hook_fn(module, input, output):
        if not isinstance(module, nn.Linear):
            return output

        ctx = getattr(module, "_mask_hook_ctx", None)
        if ctx is None or (not ctx.get("enabled", False)):
            return output

        sample = ctx["sample"]
        temb = ctx["temb"]
        res_sample = ctx["res_sample"]
        mask_generator = ctx["mask_generator"]
        adapter = ctx["adapter"]
        affine_combiner = ctx["affine_combiner"]
        encoder_hidden_states = ctx["encoder_hidden_states"]
        grounding_extra_input = ctx.get("grounding_extra_input", None)

        weight_shape = module.weight.shape
        mask = mask_generator(
            weight_shape,
            sample,
            res_sample,
            temb,
            encoder_hidden_states,
            grounding_extra_input=grounding_extra_input,
            layer_name=layer_name,
        ).to(module.weight.device)

        mask = adapter(mask)

        weight = module.weight.unsqueeze(0).expand(mask.shape[0], *module.weight.shape)
        if affine_combiner is not None:
            masked_weight = affine_combiner(weight, mask)
        else:
            masked_weight = weight * mask

        x = input[0]
        if x.dim() == 2:
            x_batched = x.unsqueeze(1)
            out = th.bmm(x_batched, masked_weight.transpose(1, 2)).squeeze(1)
        elif x.dim() == 3:
            out = th.bmm(x, masked_weight.transpose(1, 2))
        else:
            raise ValueError(
                f"Linear hook only supports input dim 2 or 3, got {x.dim()} for layer {layer_name}"
            )

        if module.bias is not None:
            if out.dim() == 2:
                out = out + module.bias.unsqueeze(0)
            else:
                out = out + module.bias.view(1, 1, -1)

        return out

    return hook_fn


def get_conv_hook_fn(layer_name=None):
    def hook_fn(module, input, output):
        if not isinstance(module, nn.Conv2d):
            return output

        ctx = getattr(module, "_mask_hook_ctx", None)
        if ctx is None or (not ctx.get("enabled", False)):
            return output

        if module.groups != 1:
            return output

        sample = ctx["sample"]
        temb = ctx["temb"]
        res_sample = ctx["res_sample"]
        mask_generator = ctx["mask_generator"]
        adapter = ctx["adapter"]
        affine_combiner = ctx["affine_combiner"]
        encoder_hidden_states = ctx["encoder_hidden_states"]
        grounding_extra_input = ctx.get("grounding_extra_input", None)

        x = input[0]
        batch_size = x.size(0)
        weight_shape = module.weight.shape

        mask = mask_generator(
            weight_shape,
            sample,
            res_sample,
            temb,
            encoder_hidden_states,
            grounding_extra_input=grounding_extra_input,
            layer_name=layer_name,
        ).to(module.weight.device)

        mask = adapter(mask)

        # 修复 1x1 conv 的 mask 维度问题
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        elif mask.dim() != 5:
            raise ValueError(
                f"Conv mask must be 3D or 5D, got {mask.dim()}D for layer {layer_name}, shape={tuple(mask.shape)}"
            )

        weight = module.weight.unsqueeze(0).expand(mask.shape[0], *module.weight.shape)
        if affine_combiner is not None:
            masked_weight = affine_combiner(weight, mask)
        else:
            masked_weight = weight * mask
        masked_weight = masked_weight.reshape(-1, *module.weight.shape[1:]).contiguous()

        input_reshaped = x.reshape(1, -1, *x.shape[2:]).contiguous()

        if module.bias is not None:
            masked_bias = module.bias.repeat(batch_size).contiguous()
        else:
            masked_bias = None

        out = F.conv2d(
            input_reshaped,
            masked_weight,
            bias=masked_bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=batch_size
        )
        out = out.reshape(batch_size, -1, *out.shape[2:]).contiguous()
        return out

    return hook_fn

###########################################################



class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        transformer_depth=1,
        context_dim=None,
        fuser_type=None,
        inpaint_mode=False,
        grounding_downsampler=None,
        grounding_tokenizer=None,
        use_mask_modules: bool = True,
        build_mask_modules: bool = True,
        enable_linear_mask: bool = True,
        enable_conv_mask: bool = False,
        enable_affine_combiner: bool = True,
        debug_mask_build=False,
        debug_mask_detail=False,
        mask_hidden_dim=64,
        mask_base_factor=4,
        sem_in_dim=152,
        use_sem_cond=True,
        use_sem_cond_for_linear_mask=None,
        use_sem_cond_for_conv_mask=None,
        skip_linear_layers=None,
        linear_mask_targets=None,
        conv_mask_targets=None,
        conv_mask_skip_1x1=True,
        conv_mask_skip_groups_not1=True,
        conv_mask_include_resblock_skip=False,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        assert fuser_type in ["gatedSA", "gatedSA2", "gatedCA"]

        self.grounding_tokenizer_input = None

        self.build_mask_modules = build_mask_modules
        self.use_mask_modules = use_mask_modules
        self.enable_linear_mask = enable_linear_mask
        self.enable_conv_mask = enable_conv_mask
        self.enable_affine_combiner = enable_affine_combiner
        self.debug_mask_build = debug_mask_build
        self.debug_mask_detail = debug_mask_detail
        self.mask_hidden_dim = mask_hidden_dim
        self.mask_base_factor = mask_base_factor
        self.sem_in_dim = sem_in_dim
        self.use_sem_cond = use_sem_cond
        self.use_sem_cond_for_linear_mask = (
            self.use_sem_cond if use_sem_cond_for_linear_mask is None else bool(use_sem_cond_for_linear_mask)
        )
        self.use_sem_cond_for_conv_mask = (
            self.use_sem_cond if use_sem_cond_for_conv_mask is None else bool(use_sem_cond_for_conv_mask)
        )
        self.skip_linear_layers = skip_linear_layers or ["time_emb_proj", "ff", "conv_shortcut", "proj_in", "proj_out"]
        self.linear_mask_targets = linear_mask_targets or ["SpatialTransformer"]
        self.conv_mask_targets = conv_mask_targets or ["ResBlock"]
        self.conv_mask_skip_1x1 = conv_mask_skip_1x1
        self.conv_mask_skip_groups_not1 = conv_mask_skip_groups_not1
        self.conv_mask_include_resblock_skip = conv_mask_include_resblock_skip

        self.linear_mask_generators = nn.ModuleList()
        self.linear_adapters = nn.ModuleList()
        self.linear_affine_combiners = nn.ModuleList()
        self.conv_mask_generators = nn.ModuleList()
        self.conv_adapters = nn.ModuleList()
        self.conv_affine_combiners = nn.ModuleList()
        self.mask_stage_meta = []
        self._mask_stage_lookup = {}
        self._hooks = []
        self._hooked_modules = []

        self._collect_gated_sa_maps = False
        self._output_gated_sa_modules = []

        def _rank0_print(*args, **kwargs):
            if th.distributed.is_available() and th.distributed.is_initialized():
                if th.distributed.get_rank() != 0:
                    return
            print(*args, **kwargs)

        self._rank0_print = _rank0_print

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_net = None
        self.additional_channel_from_downsampler = 0
        self.first_conv_type = "SD"
        self.first_conv_restorable = True
        if grounding_downsampler is not None:
            self.downsample_net = instantiate_from_config(grounding_downsampler)
            self.additional_channel_from_downsampler = self.downsample_net.out_dim
            self.first_conv_type = "GLIGEN"

        if inpaint_mode:
            in_c = in_channels + self.additional_channel_from_downsampler + in_channels + 1
            self.first_conv_restorable = False
        else:
            in_c = in_channels + self.additional_channel_from_downsampler

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_c, model_channels, 3, padding=1))]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch,
                            key_dim=context_dim,
                            value_dim=context_dim,
                            n_heads=num_heads,
                            d_head=dim_head,
                            depth=transformer_depth,
                            fuser_type=fuser_type,
                            use_checkpoint=use_checkpoint,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        dim_head = ch // num_heads

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                key_dim=context_dim,
                value_dim=context_dim,
                n_heads=num_heads,
                d_head=dim_head,
                depth=transformer_depth,
                fuser_type=fuser_type,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch,
                            key_dim=context_dim,
                            value_dim=context_dim,
                            n_heads=num_heads,
                            d_head=dim_head,
                            depth=transformer_depth,
                            fuser_type=fuser_type,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        self.position_net = instantiate_from_config(grounding_tokenizer)
        self._build_output_gated_sa_registry()

        if self.build_mask_modules:
            self._build_mask_modules(time_embed_dim)
        else:
            if self.debug_mask_build:
                self._rank0_print("[MaskBuild] build_mask_modules=False, skip building all mask modules.")

        self._register_persistent_mask_hooks()
        self._disable_checkpoint_for_masked_resblocks_in_output()

    def _iter_mask_stage_specs(self):
        specs = []
        for bi, block in enumerate(self.input_blocks):
            specs.append({"stage_group": "input", "stage_name": f"input_blocks.{bi}", "module": block, "use_res_sample": False})
        specs.append({"stage_group": "middle", "stage_name": "middle_block", "module": self.middle_block, "use_res_sample": False})
        for bi, block in enumerate(self.output_blocks):
            specs.append({"stage_group": "output", "stage_name": f"output_blocks.{bi}", "module": block, "use_res_sample": True})
        return specs

    def _get_top_level_child(self, block, name):
        if name is None or name == "":
            return None, ""
        parts = name.split('.')
        head = parts[0]
        if not head.isdigit():
            return None, name
        idx = int(head)
        if idx >= len(block):
            return None, '.'.join(parts[1:])
        return block[idx], '.'.join(parts[1:])

    def _infer_stage_channels(self, block):
        first_resblock = None
        for layer in block:
            if isinstance(layer, ResBlock):
                first_resblock = layer
                break
        if first_resblock is None:
            return None, None
        return first_resblock.channels, first_resblock.out_channels

    def _belongs_to_resblock_conv(self, sub_name):
        if sub_name.endswith("in_layers.2") or sub_name.endswith("out_layers.3"):
            return True
        if self.conv_mask_include_resblock_skip and "skip_connection" in sub_name:
            return True
        return False

    def _belongs_to_spatial_transformer_conv(self, sub_name):
        return (sub_name == "proj_in") or (sub_name == "proj_out")

    def _belongs_to_upsample_conv(self, sub_name):
        return sub_name == "conv"

    def _should_mask_linear(self, block, name, module):
        if not isinstance(module, nn.Linear):
            return False
        if any(s in name for s in self.skip_linear_layers):
            return False
        top_module, _ = self._get_top_level_child(block, name)
        if top_module is None:
            return False

        hit = False
        if "ResBlock" in self.linear_mask_targets:
            hit = hit or isinstance(top_module, ResBlock)
        if "SpatialTransformer" in self.linear_mask_targets:
            hit = hit or isinstance(top_module, SpatialTransformer)
        if "Upsample" in self.linear_mask_targets:
            hit = hit or isinstance(top_module, Upsample)
        return hit

    def _should_mask_conv(self, block, name, module):
        if not isinstance(module, nn.Conv2d):
            return False
        if self.conv_mask_skip_groups_not1 and module.groups != 1:
            return False
        if self.conv_mask_skip_1x1 and module.kernel_size == (1, 1):
            return False

        top_module, sub_name = self._get_top_level_child(block, name)
        if top_module is None:
            return False

        hit = False
        if "ResBlock" in self.conv_mask_targets and isinstance(top_module, ResBlock):
            hit = hit or self._belongs_to_resblock_conv(sub_name)
        if "SpatialTransformer" in self.conv_mask_targets and isinstance(top_module, SpatialTransformer):
            hit = hit or self._belongs_to_spatial_transformer_conv(sub_name)
        if "Upsample" in self.conv_mask_targets and isinstance(top_module, Upsample):
            hit = hit or self._belongs_to_upsample_conv(sub_name)
        return hit

    def _build_mask_modules(self, time_embed_dim):
        base_factor = self.mask_base_factor
        if self.debug_mask_build:
            self._rank0_print(
                f"[MaskBuild] stages=input+middle+output | linear_targets={self.linear_mask_targets} | "
                f"conv_targets={self.conv_mask_targets} | base_factor={base_factor} | "
                f"enable_linear_mask={self.enable_linear_mask} | enable_conv_mask={self.enable_conv_mask} | "
                f"enable_affine_combiner={self.enable_affine_combiner} | "
                f"use_sem_cond(linear={self.use_sem_cond_for_linear_mask}, conv={self.use_sem_cond_for_conv_mask}, default={self.use_sem_cond})"
            )

        total_linear = 0
        total_linear_skipped = 0
        total_linear_hooked = 0
        total_conv = 0
        total_conv_skipped = 0
        total_conv_hooked = 0

        for stage_idx, spec in enumerate(self._iter_mask_stage_specs()):
            block = spec["module"]
            in_c, out_c = self._infer_stage_channels(block)
            if in_c is None or out_c is None:
                if self.debug_mask_build:
                    self._rank0_print(f"[MaskBuild][{stage_idx:02d}][{spec['stage_name']}] no ResBlock found, skip.")
                continue

            f = base_factor
            while f > 1 and (in_c % f != 0 or out_c % f != 0):
                f //= 2
            if f < 1:
                f = 1

            meta = {
                "stage_idx": stage_idx,
                "stage_name": spec["stage_name"],
                "stage_group": spec["stage_group"],
                "module": block,
                "use_res_sample": spec["use_res_sample"],
                "in_c": in_c,
                "out_c": out_c,
                "factor": f,
            }
            self.mask_stage_meta.append(meta)
            self._mask_stage_lookup[id(block)] = len(self.mask_stage_meta) - 1

            if self.enable_linear_mask:
                mg_linear = DecoderWeightsMaskGenerator(
                    in_c, out_c, hidden_dim=self.mask_hidden_dim, factor=f,
                    temb_dim=time_embed_dim, sem_in_dim=self.sem_in_dim,
                    use_sem_cond=self.use_sem_cond_for_linear_mask,
                )
                ad_linear = nn.ModuleDict()
                affine_linear = nn.ModuleDict()
                linear_total = 0
                linear_skipped = 0
                for name, subm in block.named_modules():
                    if isinstance(subm, nn.Linear):
                        linear_total += 1
                        if not self._should_mask_linear(block, name, subm):
                            linear_skipped += 1
                            continue
                        k = name.replace('.', '_')
                        ad_linear[k] = Adapter(
                            out_c // f, in_c // f, subm.out_features, subm.in_features,
                            tau=1.0,
                            init_bias=0.0,
                            init_weight_std=1e-3,
                        )
                        if self.enable_affine_combiner:
                            affine_linear[k] = AffineWeightCombiner(init_k0=1.0, init_k1=0.0, init_k2=1e-2)
                self.linear_mask_generators.append(mg_linear)
                self.linear_adapters.append(ad_linear)
                self.linear_affine_combiners.append(affine_linear)
            else:
                linear_total = 0
                linear_skipped = 0
                ad_linear = nn.ModuleDict()
                affine_linear = nn.ModuleDict()
                self.linear_mask_generators.append(nn.Identity())
                self.linear_adapters.append(ad_linear)
                self.linear_affine_combiners.append(affine_linear)

            if self.enable_conv_mask:
                mg_conv = DecoderWeightsMaskGenerator(
                    in_c, out_c, hidden_dim=self.mask_hidden_dim, factor=f,
                    temb_dim=time_embed_dim, sem_in_dim=self.sem_in_dim,
                    use_sem_cond=self.use_sem_cond_for_conv_mask,
                )
                ad_conv = nn.ModuleDict()
                affine_conv = nn.ModuleDict()
                conv_total = 0
                conv_skipped = 0
                for name, subm in block.named_modules():
                    if isinstance(subm, nn.Conv2d):
                        conv_total += 1
                        if not self._should_mask_conv(block, name, subm):
                            conv_skipped += 1
                            continue
                        k = name.replace('.', '_')
                        ad_conv[k] = Adapter(
                            out_c // f, in_c // f, subm.out_channels, subm.in_channels,
                            tau=1.0,
                            init_bias=0.0,
                            init_weight_std=1e-3,
                        )
                        if self.enable_affine_combiner:
                            affine_conv[k] = AffineWeightCombiner(init_k0=1.0, init_k1=0.0, init_k2=1e-2)
                self.conv_mask_generators.append(mg_conv)
                self.conv_adapters.append(ad_conv)
                self.conv_affine_combiners.append(affine_conv)
            else:
                conv_total = 0
                conv_skipped = 0
                ad_conv = nn.ModuleDict()
                affine_conv = nn.ModuleDict()
                self.conv_mask_generators.append(nn.Identity())
                self.conv_adapters.append(ad_conv)
                self.conv_affine_combiners.append(affine_conv)

            if self.debug_mask_build:
                linear_mg_params = sum(p.numel() for p in self.linear_mask_generators[-1].parameters()) if self.enable_linear_mask else 0
                linear_ad_params = sum(p.numel() for p in ad_linear.parameters())
                linear_affine_params = sum(p.numel() for p in affine_linear.parameters())
                conv_mg_params = sum(p.numel() for p in self.conv_mask_generators[-1].parameters()) if self.enable_conv_mask else 0
                conv_ad_params = sum(p.numel() for p in ad_conv.parameters())
                conv_affine_params = sum(p.numel() for p in affine_conv.parameters())
                self._rank0_print(
                    f"[MaskBuild][{len(self.mask_stage_meta)-1:02d}][{spec['stage_name']}] in_c={in_c} out_c={out_c} factor={f} | "
                    f"Linear(total={linear_total}, skipped={linear_skipped}, hooked={len(ad_linear)}) "
                    f"mg_params={linear_mg_params} ad_params={linear_ad_params} affine_params={linear_affine_params} | "
                    f"Conv(total={conv_total}, skipped={conv_skipped}, hooked={len(ad_conv)}) "
                    f"mg_params={conv_mg_params} ad_params={conv_ad_params} affine_params={conv_affine_params}"
                )
                if self.debug_mask_detail:
                    self._rank0_print(f"  linear adapter keys sample: {list(ad_linear.keys())[:10]}")
                    self._rank0_print(f"  conv adapter keys sample: {list(ad_conv.keys())[:10]}")

            total_linear += linear_total
            total_linear_skipped += linear_skipped
            total_linear_hooked += len(ad_linear)
            total_conv += conv_total
            total_conv_skipped += conv_skipped
            total_conv_hooked += len(ad_conv)

        if self.debug_mask_build:
            self._rank0_print(
                f"[MaskBuild][SUM] stages={len(self.mask_stage_meta)} | "
                f"Linear(total={total_linear}, skipped={total_linear_skipped}, hooked={total_linear_hooked}) | "
                f"Conv(total={total_conv}, skipped={total_conv_skipped}, hooked={total_conv_hooked})"
            )
            self._rank0_print(
                f"[MaskBuild][TOTAL] linear_mask_generators={len(self.linear_mask_generators)}, "
                f"linear_adapters={sum(len(d) for d in self.linear_adapters)}, "
                f"linear_affine_combiners={sum(len(d) for d in self.linear_affine_combiners)} | "
                f"conv_mask_generators={len(self.conv_mask_generators)}, "
                f"conv_adapters={sum(len(d) for d in self.conv_adapters)}, "
                f"conv_affine_combiners={sum(len(d) for d in self.conv_affine_combiners)}"
            )

    def _disable_checkpoint_for_masked_resblocks_in_output(self):
        return

    def _build_output_gated_sa_registry(self):
        self._output_gated_sa_modules = []
        for bi, block in enumerate(self.output_blocks):
            for name, sub_module in block.named_modules():
                if isinstance(sub_module, SpatialTransformer):
                    for ti, tb in enumerate(sub_module.transformer_blocks):
                        fuser = getattr(tb, "fuser", None)
                        if isinstance(fuser, (GatedSelfAttentionDense, GatedSelfAttentionDense2)):
                            if hasattr(tb, "use_checkpoint"):
                                tb.use_checkpoint = False
                            if hasattr(fuser, "capture_attention"):
                                fuser.capture_attention = False
                            if hasattr(fuser, "reset_attention_cache"):
                                fuser.reset_attention_cache()
                            self._output_gated_sa_modules.append({
                                "name": f"output_blocks.{bi}.{name}.transformer_blocks.{ti}.fuser",
                                "module": fuser,
                            })
        if self.debug_mask_build:
            self._rank0_print(f"[AttnBuild] output_gated_sa_modules={len(self._output_gated_sa_modules)}")

    def set_attention_map_collection(self, enabled: bool):
        self._collect_gated_sa_maps = bool(enabled)
        for item in self._output_gated_sa_modules:
            module = item["module"]
            if hasattr(module, "capture_attention"):
                module.capture_attention = bool(enabled)
            if not enabled and hasattr(module, "reset_attention_cache"):
                module.reset_attention_cache()

    def clear_attention_map_cache(self):
        for item in self._output_gated_sa_modules:
            module = item["module"]
            if hasattr(module, "reset_attention_cache"):
                module.reset_attention_cache()

    def get_collected_gated_sa_maps(self):
        records = []
        for item in self._output_gated_sa_modules:
            module = item["module"]
            attn_map = getattr(module, "last_attention_map", None)
            meta = getattr(module, "last_attention_meta", None)
            if attn_map is None:
                continue
            record = {"name": item["name"], "attn": attn_map}
            if isinstance(meta, dict):
                record.update(meta)
            records.append(record)
        return records

    def _remove_all_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _clear_hooks(self):
        for m in self._hooked_modules:
            try:
                m._mask_hook_ctx = None
            except Exception:
                pass

    def _register_persistent_mask_hooks(self):
        self._remove_all_hooks()
        self._hooked_modules = []

        if not self.build_mask_modules:
            return

        linear_cnt = 0
        conv_cnt = 0

        for stage_idx, meta in enumerate(self.mask_stage_meta):
            block = meta["module"]
            if self.enable_linear_mask and stage_idx < len(self.linear_adapters) and stage_idx < len(self.linear_affine_combiners):
                adapters = self.linear_adapters[stage_idx]
                affine_combiners = self.linear_affine_combiners[stage_idx]
                for name, sub_module in block.named_modules():
                    if not isinstance(sub_module, nn.Linear):
                        continue
                    if not self._should_mask_linear(block, name, sub_module):
                        continue
                    k = name.replace('.', '_')
                    if k not in adapters:
                        continue
                    if self.enable_affine_combiner and k not in affine_combiners:
                        continue
                    sub_module._mask_hook_ctx = None
                    hook = sub_module.register_forward_hook(get_linear_hook_fn(layer_name=f"{meta['stage_name']}.{name}"))
                    self._hooks.append(hook)
                    self._hooked_modules.append(sub_module)
                    linear_cnt += 1

            if self.enable_conv_mask and stage_idx < len(self.conv_adapters) and stage_idx < len(self.conv_affine_combiners):
                adapters = self.conv_adapters[stage_idx]
                affine_combiners = self.conv_affine_combiners[stage_idx]
                for name, sub_module in block.named_modules():
                    if not isinstance(sub_module, nn.Conv2d):
                        continue
                    if not self._should_mask_conv(block, name, sub_module):
                        continue
                    k = name.replace('.', '_')
                    if k not in adapters:
                        continue
                    if self.enable_affine_combiner and k not in affine_combiners:
                        continue
                    sub_module._mask_hook_ctx = None
                    hook = sub_module.register_forward_hook(get_conv_hook_fn(layer_name=f"{meta['stage_name']}.{name}"))
                    self._hooks.append(hook)
                    self._hooked_modules.append(sub_module)
                    conv_cnt += 1

        if self.debug_mask_build:
            self._rank0_print(f"[MaskBuild][PersistentHooks] linear={linear_cnt}, conv={conv_cnt}")

    def _register_linear_hooks(
        self,
        module,
        sample,
        res_sample,
        temb,
        mask_generator,
        adapters,
        affine_combiners,
        context,
        grounding_extra_input=None,
    ):
        for name, sub_module in module.named_modules():
            if not isinstance(sub_module, nn.Linear):
                continue
            if name.replace('.', '_') not in adapters:
                continue
            affine_combiner = None
            if self.enable_affine_combiner:
                if name.replace('.', '_') not in affine_combiners:
                    continue
                affine_combiner = affine_combiners[name.replace('.', '_')]
            sub_module._mask_hook_ctx = {
                "enabled": True,
                "sample": sample,
                "temb": temb,
                "res_sample": res_sample,
                "mask_generator": mask_generator,
                "adapter": adapters[name.replace('.', '_')],
                "affine_combiner": affine_combiner,
                "encoder_hidden_states": context,
                "grounding_extra_input": grounding_extra_input,
            }

    def _register_conv_hooks(
        self,
        module,
        sample,
        res_sample,
        temb,
        mask_generator,
        adapters,
        affine_combiners,
        context,
        grounding_extra_input=None,
    ):
        for name, sub_module in module.named_modules():
            if not isinstance(sub_module, nn.Conv2d):
                continue
            if name.replace('.', '_') not in adapters:
                continue
            affine_combiner = None
            if self.enable_affine_combiner:
                if name.replace('.', '_') not in affine_combiners:
                    continue
                affine_combiner = affine_combiners[name.replace('.', '_')]
            sub_module._mask_hook_ctx = {
                "enabled": True,
                "sample": sample,
                "temb": temb,
                "res_sample": res_sample,
                "mask_generator": mask_generator,
                "adapter": adapters[name.replace('.', '_')],
                "affine_combiner": affine_combiner,
                "encoder_hidden_states": context,
                "grounding_extra_input": grounding_extra_input,
            }

    def _activate_mask_hooks_for_stage(self, stage_idx, sample, res_sample, temb, context, grounding_extra_input=None):
        if stage_idx is None:
            return
        if self.enable_linear_mask:
            self._register_linear_hooks(
                module=self.mask_stage_meta[stage_idx]["module"],
                sample=sample,
                res_sample=res_sample,
                temb=temb,
                mask_generator=self.linear_mask_generators[stage_idx],
                adapters=self.linear_adapters[stage_idx],
                affine_combiners=self.linear_affine_combiners[stage_idx],
                context=context,
                grounding_extra_input=(grounding_extra_input if self.use_sem_cond_for_linear_mask else None),
            )
        if self.enable_conv_mask:
            self._register_conv_hooks(
                module=self.mask_stage_meta[stage_idx]["module"],
                sample=sample,
                res_sample=res_sample,
                temb=temb,
                mask_generator=self.conv_mask_generators[stage_idx],
                adapters=self.conv_adapters[stage_idx],
                affine_combiners=self.conv_affine_combiners[stage_idx],
                context=context,
                grounding_extra_input=(grounding_extra_input if self.use_sem_cond_for_conv_mask else None),
            )

    def _prepare_grounding_extra_for_hook(self, grounding_extra_input, sample, sem_cache):
        if grounding_extra_input is None:
            return None
        if grounding_extra_input.dim() != 4:
            raise ValueError(f"grounding_extra_input must be 4D [B,C,H,W], got {tuple(grounding_extra_input.shape)}")
        if sample.dim() != 4:
            raise ValueError(f"sample must be 4D [B,C,H,W], got {tuple(sample.shape)}")

        target_hw = (sample.shape[-2], sample.shape[-1])
        sem0 = grounding_extra_input.to(device=sample.device)
        if not th.is_floating_point(sem0):
            sem0 = sem0.float()
        sem0 = sem0.to(dtype=sample.dtype)

        if tuple(sem0.shape[-2:]) == tuple(target_hw):
            return sem0
        if target_hw not in sem_cache:
            sem_cache[target_hw] = F.interpolate(sem0, size=target_hw, mode="nearest")
        return sem_cache[target_hw]

    def restore_first_conv_from_SD(self):
        if self.first_conv_restorable:
            device = self.input_blocks[0][0].weight.device
            SD_weights = th.load("SD_input_conv_weight_bias.pth")
            self.GLIGEN_first_conv_state_dict = deepcopy(self.input_blocks[0][0].state_dict())
            self.input_blocks[0][0] = conv_nd(2, 4, 320, 3, padding=1)
            self.input_blocks[0][0].load_state_dict(SD_weights)
            self.input_blocks[0][0].to(device)
            self.first_conv_type = "SD"
        else:
            print("First conv layer is not restorable and skipped this process, probably because this is an inpainting model?")

    def restore_first_conv_from_GLIGEN(self):
        breakpoint()

    def set_all_mask_adapters_hard(self, hard: bool):
        for ad_dict in self.linear_adapters:
            if isinstance(ad_dict, nn.ModuleDict):
                for m in ad_dict.values():
                    if hasattr(m, "set_hard"):
                        m.set_hard(hard)
        for ad_dict in self.conv_adapters:
            if isinstance(ad_dict, nn.ModuleDict):
                for m in ad_dict.values():
                    if hasattr(m, "set_hard"):
                        m.set_hard(hard)
        self._rank0_print(f"[MaskAdapter] set hard={hard}")

    def forward(self, input):
        if ("grounding_input" in input):
            grounding_input = input["grounding_input"]
        else:
            grounding_input = self.grounding_tokenizer_input.get_null_input()

        if self.training and random.random() < 0.1 and self.grounding_tokenizer_input.set:
            grounding_input = self.grounding_tokenizer_input.get_null_input()

        objs = self.position_net(**grounding_input)

        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        grounding_extra_input = input.get("grounding_extra_input", None)

        h = input["x"]
        if self.downsample_net is not None and self.first_conv_type == "GLIGEN":
            temp = self.downsample_net(grounding_extra_input)
            h = th.cat([h, temp], dim=1)

        if self.inpaint_mode:
            if self.downsample_net is not None:
                breakpoint()
            h = th.cat([h, input["inpainting_extra_input"]], dim=1)

        context = input["context"]

        if getattr(self, "debug_mask_build", False) and not hasattr(self, "_debug_forward_once"):
            self._debug_forward_once = True
            self._rank0_print(
                f"[MaskForward] t_emb={tuple(emb.shape)} context={tuple(context.shape)} "
                f"grounding_extra_input={None if grounding_extra_input is None else tuple(grounding_extra_input.shape)}"
            )

        use_linear_mask = (
            self.use_mask_modules and self.build_mask_modules and self.enable_linear_mask and len(self.linear_mask_generators) == len(self.mask_stage_meta)
        )
        use_conv_mask = (
            self.use_mask_modules and self.build_mask_modules and self.enable_conv_mask and len(self.conv_mask_generators) == len(self.mask_stage_meta)
        )
        need_sem_for_hook = (
            (use_linear_mask and self.use_sem_cond_for_linear_mask) or
            (use_conv_mask and self.use_sem_cond_for_conv_mask)
        )

        self._clear_hooks()
        if self._collect_gated_sa_maps:
            self.clear_attention_map_cache()
        sem_cache = {}
        hs = []

        for module in self.input_blocks:
            stage_idx = self._mask_stage_lookup.get(id(module), None)
            grounding_extra_for_hook = None
            if stage_idx is not None and need_sem_for_hook:
                grounding_extra_for_hook = self._prepare_grounding_extra_for_hook(grounding_extra_input, h, sem_cache)
            if stage_idx is not None and (use_linear_mask or use_conv_mask):
                self._activate_mask_hooks_for_stage(stage_idx, h, None, emb, context, grounding_extra_for_hook)
            h = module(h, emb, context, objs)
            hs.append(h)

        stage_idx = self._mask_stage_lookup.get(id(self.middle_block), None)
        grounding_extra_for_hook = None
        if stage_idx is not None and need_sem_for_hook:
            grounding_extra_for_hook = self._prepare_grounding_extra_for_hook(grounding_extra_input, h, sem_cache)
        if stage_idx is not None and (use_linear_mask or use_conv_mask):
            self._activate_mask_hooks_for_stage(stage_idx, h, None, emb, context, grounding_extra_for_hook)
        h = self.middle_block(h, emb, context, objs)

        for module in self.output_blocks:
            res = hs.pop()
            stage_idx = self._mask_stage_lookup.get(id(module), None)
            grounding_extra_for_hook = None
            if stage_idx is not None and need_sem_for_hook:
                grounding_extra_for_hook = self._prepare_grounding_extra_for_hook(grounding_extra_input, h, sem_cache)
            if stage_idx is not None and (use_linear_mask or use_conv_mask):
                self._activate_mask_hooks_for_stage(stage_idx, h, res, emb, context, grounding_extra_for_hook)
            h = th.cat([h, res], dim=1)
            h = module(h, emb, context, objs)

        return self.out(h)

