# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import Tensor

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)



def extract_high_frequency_map(images, mode="sobel", normalize=True, eps=1e-6):
    if images is None:
        return None
    if images.dim() != 4:
        raise ValueError(f"images must be 4D [B,C,H,W], got {tuple(images.shape)}")

    if images.shape[1] == 1:
        gray = images
    elif images.shape[1] >= 3:
        weights = images.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        gray = (images[:, :3] * weights).sum(dim=1, keepdim=True)
    else:
        gray = images.mean(dim=1, keepdim=True)

    gray = gray.float()

    if mode == "sobel":
        kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        ky = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(gray_pad, kx)
        gy = F.conv2d(gray_pad, ky)
        hf = torch.cat([gx, gy], dim=1)
    elif mode == "laplacian":
        k = gray.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        lap = F.conv2d(gray_pad, k)
        hf = torch.cat([lap, lap.abs()], dim=1)
    else:
        raise ValueError(f"Unsupported high-frequency mode: {mode}")

    if normalize:
        denom = hf.abs().mean(dim=(2, 3), keepdim=True).clamp_min(eps)
        hf = hf / denom
    return hf.to(dtype=images.dtype, device=images.device)


class DecoderWeightsMaskGenerator(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        factor=16,
        highfreq_in_dim=2,
        use_highfreq_cond=True,
    ):
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
            nn.ReLU(inplace=True),
        )

        self.use_highfreq_cond = use_highfreq_cond
        self.highfreq_in_dim = highfreq_in_dim
        if self.use_highfreq_cond:
            self.proj_highfreq = nn.Conv2d(highfreq_in_dim, in_channels, kernel_size=1, bias=False)
            nn.init.constant_(self.proj_highfreq.weight, 0.0)

        self.in_c = new_in
        self.out_c = new_out
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1, 1) and m.out_channels == self.in_c * self.out_c:
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            else:
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, weight_shape, sample, highfreq_cond=None, layer_name=None):
        flag = len(weight_shape)
        x = sample

        if flag == 4:
            _, _, k_h, k_w = weight_shape
            x = F.adaptive_avg_pool2d(x, (k_h, k_w))
            target_hw = (k_h, k_w)
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            target_hw = (1, 1)

        hf_cond = 0.0
        if self.use_highfreq_cond and highfreq_cond is not None:
            if highfreq_cond.dim() != 4:
                raise ValueError(
                    f"highfreq_cond must be 4D [B,C,H,W], got {tuple(highfreq_cond.shape)} for layer {layer_name}"
                )
            if highfreq_cond.shape[1] != self.highfreq_in_dim:
                raise ValueError(
                    f"highfreq_cond channel mismatch: expect {self.highfreq_in_dim}, got {highfreq_cond.shape[1]} for layer {layer_name}"
                )
            if tuple(highfreq_cond.shape[-2:]) != tuple(target_hw):
                highfreq_cond = F.adaptive_avg_pool2d(highfreq_cond, target_hw)
            hf_cond = self.proj_highfreq(highfreq_cond).to(dtype=x.dtype, device=x.device)

        x = x + hf_cond
        mask = self.conv_kernel_mask(x)

        if flag == 4:
            mask = mask.view(mask.size(0), self.out_c, self.in_c, k_h, k_w)
        else:
            mask = mask.view(mask.size(0), self.out_c, self.in_c)
        return mask


def gumbel_sigmoid(logits: Tensor, tau: float = 1.0, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()
    if hard:
        y_hard = (y_soft > threshold).to(y_soft.dtype)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class Adapter(nn.Module):
    def __init__(
        self,
        out_c,
        in_c,
        new_out_c,
        new_in_c,
        tau=1.0,
        init_bias=0.0,
        init_weight_std=1e-3,
        hard=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(out_c, new_out_c, kernel_size=1)
        self.conv_out = nn.Conv2d(in_c, new_in_c, kernel_size=1)
        self.tau = tau
        self.init_bias = init_bias
        self.init_weight_std = init_weight_std
        self.hard = hard
        self.apply(self.weights_init)

    def set_hard(self, hard: bool):
        self.hard = bool(hard)

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
        x = gumbel_sigmoid(x, tau=self.tau, hard=self.hard)
        x = rearrange(x, '(b new_out_c) new_in_c h w -> b new_out_c new_in_c h w', b=input_tensor.shape[0])

        if x.size(-1) == 1 and x.size(-2) == 1:
            x = x.squeeze(-1).squeeze(-1)
        return x


class AffineWeightCombiner(nn.Module):
    def __init__(self, init_k0=1.0, init_k1=0.0, init_k2=1e-2):
        super().__init__()
        self.k0 = nn.Parameter(torch.tensor(float(init_k0)))
        self.k1 = nn.Parameter(torch.tensor(float(init_k1)))
        self.k2 = nn.Parameter(torch.tensor(float(init_k2)))

    def forward(self, weight, mask):
        if weight.dim() != mask.dim():
            raise ValueError(
                f"AffineWeightCombiner expects weight and mask with same dims, got weight.dim={weight.dim()} mask.dim={mask.dim()}"
            )
        ones = torch.ones_like(weight)
        return self.k0 * weight + self.k1 * ones + self.k2 * mask


def get_decoder_conv_hook_fn(layer_name=None):
    def hook_fn(module, input, output):
        if not isinstance(module, nn.Conv2d):
            return output

        ctx = getattr(module, "_mask_hook_ctx", None)
        if ctx is None or (not ctx.get("enabled", False)):
            return output
        if module.groups != 1:
            return output

        sample = ctx["sample"]
        mask_generator = ctx["mask_generator"]
        adapter = ctx["adapter"]
        affine_combiner = ctx["affine_combiner"]
        highfreq_cond = ctx.get("highfreq_cond", None)

        x = input[0]
        batch_size = x.size(0)
        if highfreq_cond is not None and highfreq_cond.size(0) != batch_size:
            if highfreq_cond.size(0) == 1:
                highfreq_cond = highfreq_cond.expand(batch_size, -1, -1, -1)
            else:
                raise ValueError(
                    f"highfreq_cond batch mismatch for layer {layer_name}: cond batch={highfreq_cond.size(0)}, input batch={batch_size}"
                )

        mask = mask_generator(
            module.weight.shape,
            sample,
            highfreq_cond=highfreq_cond,
            layer_name=layer_name,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        mask = adapter(mask)

        if mask.dim() == 3:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        elif mask.dim() != 5:
            raise ValueError(
                f"Conv mask must be 3D or 5D, got {mask.dim()}D for layer {layer_name}, shape={tuple(mask.shape)}"
            )

        weight = module.weight.unsqueeze(0).expand(mask.shape[0], *module.weight.shape)
        masked_weight = affine_combiner(weight, mask) if affine_combiner is not None else (weight * mask)
        masked_weight = masked_weight.reshape(-1, *module.weight.shape[1:]).contiguous()

        input_reshaped = x.reshape(1, -1, *x.shape[2:]).contiguous()
        masked_bias = module.bias.repeat(batch_size).contiguous() if module.bias is not None else None

        out = F.conv2d(
            input_reshaped,
            masked_weight,
            bias=masked_bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=batch_size,
        )
        out = out.reshape(batch_size, -1, *out.shape[2:]).contiguous()
        return out

    return hook_fn


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        use_decoder_mask=False,
        build_decoder_mask_modules=True,
        decoder_mask_hidden_dim=64,
        decoder_mask_base_factor=4,
        decoder_mask_use_affine=True,
        decoder_enable_affine_combiner=None,
        decoder_mask_use_highfreq_cond=True,
        decoder_mask_highfreq_in_dim=2,
        decoder_mask_highfreq_mode="sobel",
        decoder_mask_skip_1x1=True,
        decoder_mask_skip_groups_not1=True,
        decoder_mask_include_shortcut=False,
        decoder_mask_include_upsample=True,
        decoder_mask_hard=True,
        debug_decoder_mask=False,
        decoder_mask_build_verbose=False,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        self.use_decoder_mask = use_decoder_mask
        self.build_decoder_mask_modules = build_decoder_mask_modules
        self.decoder_mask_hidden_dim = decoder_mask_hidden_dim
        self.decoder_mask_base_factor = decoder_mask_base_factor
        self.decoder_enable_affine_combiner = (
            bool(decoder_mask_use_affine)
            if decoder_enable_affine_combiner is None
            else bool(decoder_enable_affine_combiner)
        )
        self.decoder_mask_use_affine = self.decoder_enable_affine_combiner
        self.decoder_mask_use_highfreq_cond = decoder_mask_use_highfreq_cond
        self.decoder_mask_highfreq_in_dim = decoder_mask_highfreq_in_dim
        self.decoder_mask_highfreq_mode = decoder_mask_highfreq_mode
        self.decoder_mask_skip_1x1 = decoder_mask_skip_1x1
        self.decoder_mask_skip_groups_not1 = decoder_mask_skip_groups_not1
        self.decoder_mask_include_shortcut = decoder_mask_include_shortcut
        self.decoder_mask_include_upsample = decoder_mask_include_upsample
        self.decoder_mask_hard = decoder_mask_hard
        self.debug_decoder_mask = debug_decoder_mask
        self.decoder_mask_build_verbose = decoder_mask_build_verbose

        self._stored_mask_cond_images = None
        self._stored_mask_cond_highfreq = None
        self._mask_hooks = []
        self._hooked_modules = []
        self.decoder_mask_generators = nn.ModuleList()
        self.decoder_adapters = nn.ModuleList()
        self.decoder_affine_combiners = nn.ModuleList()
        self.decoder_mask_build_stats = []
        self.decoder_mask_build_total_hooked = 0

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        if self.build_decoder_mask_modules and self.use_decoder_mask:
            self._build_decoder_mask_modules()
            self._register_persistent_decoder_mask_hooks()

    def extract_high_frequency(self, images, mode=None):
        mode = self.decoder_mask_highfreq_mode if mode is None else mode
        return extract_high_frequency_map(images, mode=mode)

    def set_mask_reference_images(self, images, mode=None, keep_highfreq=True):
        self._stored_mask_cond_images = images
        if keep_highfreq and images is not None:
            self._stored_mask_cond_highfreq = self.extract_high_frequency(images, mode=mode)
        else:
            self._stored_mask_cond_highfreq = None

    def clear_mask_reference_images(self):
        self._stored_mask_cond_images = None
        self._stored_mask_cond_highfreq = None

    def set_all_mask_adapters_hard(self, hard: bool):
        for ad_dict in self.decoder_adapters:
            if isinstance(ad_dict, nn.ModuleDict):
                for m in ad_dict.values():
                    if hasattr(m, "set_hard"):
                        m.set_hard(hard)

    def _should_mask_decoder_conv(self, name, module):
        if not isinstance(module, nn.Conv2d):
            return False
        if self.decoder_mask_skip_groups_not1 and module.groups != 1:
            return False
        if self.decoder_mask_skip_1x1 and module.kernel_size == (1, 1):
            return False
        if name.endswith("conv1") or name.endswith("conv2"):
            return True
        if self.decoder_mask_include_shortcut and (name.endswith("conv_shortcut") or name.endswith("nin_shortcut")):
            return True
        if self.decoder_mask_include_upsample and name.endswith("upsample.conv"):
            return True
        return False

    def _build_decoder_mask_modules(self):
        base_factor = self.decoder_mask_base_factor
        total_hooked = 0
        self.decoder_mask_build_stats = []
        for bi, block in enumerate(self.up):
            first_rb = None
            for layer in block.block:
                if isinstance(layer, ResnetBlock):
                    first_rb = layer
                    break

            if first_rb is None:
                self.decoder_mask_generators.append(nn.Identity())
                self.decoder_adapters.append(nn.ModuleDict())
                self.decoder_affine_combiners.append(nn.ModuleDict())
                stats = {
                    "block_index": bi,
                    "in_c": None,
                    "out_c": None,
                    "factor": None,
                    "linear_total": 0,
                    "linear_skipped": 0,
                    "linear_hooked": 0,
                    "linear_mg_params": 0,
                    "linear_ad_params": 0,
                    "linear_affine_params": 0,
                    "conv_total": 0,
                    "conv_skipped": 0,
                    "conv_hooked": 0,
                    "conv_mg_params": 0,
                    "conv_ad_params": 0,
                    "conv_affine_params": 0,
                }
                self.decoder_mask_build_stats.append(stats)
                continue

            in_c = first_rb.in_channels
            out_c = first_rb.out_channels
            f = base_factor
            while f > 1 and (in_c % f != 0 or out_c % f != 0):
                f //= 2
            f = max(f, 1)

            mg = DecoderWeightsMaskGenerator(
                in_c,
                out_c,
                hidden_dim=self.decoder_mask_hidden_dim,
                factor=f,
                highfreq_in_dim=self.decoder_mask_highfreq_in_dim,
                use_highfreq_cond=self.decoder_mask_use_highfreq_cond,
            )
            adapters = nn.ModuleDict()
            affine_combiners = nn.ModuleDict()
            total = 0
            skipped = 0

            for name, subm in block.named_modules():
                if not isinstance(subm, nn.Conv2d):
                    continue
                total += 1
                if not self._should_mask_decoder_conv(name, subm):
                    skipped += 1
                    continue
                k = name.replace('.', '_')
                adapters[k] = Adapter(
                    out_c // f,
                    in_c // f,
                    subm.out_channels,
                    subm.in_channels,
                    tau=1.0,
                    init_bias=0.0,
                    init_weight_std=1e-3,
                    hard=self.decoder_mask_hard,
                )
                if self.decoder_mask_use_affine:
                    affine_combiners[k] = AffineWeightCombiner(init_k0=1.0, init_k1=0.0, init_k2=1e-2)

            self.decoder_mask_generators.append(mg)
            self.decoder_adapters.append(adapters)
            self.decoder_affine_combiners.append(affine_combiners)
            total_hooked += len(adapters)

            conv_mg_params = sum(p.numel() for p in mg.parameters())
            conv_ad_params = sum(p.numel() for p in adapters.parameters())
            conv_affine_params = sum(p.numel() for p in affine_combiners.parameters())
            stats = {
                "block_index": bi,
                "in_c": in_c,
                "out_c": out_c,
                "factor": f,
                "linear_total": 0,
                "linear_skipped": 0,
                "linear_hooked": 0,
                "linear_mg_params": 0,
                "linear_ad_params": 0,
                "linear_affine_params": 0,
                "conv_total": total,
                "conv_skipped": skipped,
                "conv_hooked": len(adapters),
                "conv_mg_params": conv_mg_params,
                "conv_ad_params": conv_ad_params,
                "conv_affine_params": conv_affine_params,
            }
            self.decoder_mask_build_stats.append(stats)

            if self.decoder_mask_build_verbose:
                print(
                    f"[DecoderMaskBuild][{bi:02d}] in_c={in_c} out_c={out_c} factor={f} | "
                    f"Linear(total=0, skipped=0, hooked=0) mg_params=0 ad_params=0 affine_params=0 | "
                    f"Conv(total={total}, skipped={skipped}, hooked={len(adapters)}) "
                    f"mg_params={conv_mg_params} ad_params={conv_ad_params} affine_params={conv_affine_params}"
                )
            elif self.debug_decoder_mask:
                print(
                    f"[DecoderMaskBuild][{bi:02d}] in_c={in_c} out_c={out_c} factor={f} total_conv={total} skipped={skipped} hooked={len(adapters)}"
                )

        self.decoder_mask_build_total_hooked = total_hooked
        if self.debug_decoder_mask or self.decoder_mask_build_verbose:
            print(
                f"[DecoderMaskBuild][SUM] blocks={len(self.up)} hooked={total_hooked} "
                f"use_highfreq={self.decoder_mask_use_highfreq_cond} use_affine={self.decoder_mask_use_affine}"
            )

    def get_decoder_mask_build_stats(self):
        return list(self.decoder_mask_build_stats)

    def _remove_all_decoder_mask_hooks(self):
        for h in self._mask_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._mask_hooks = []

    def _clear_decoder_mask_ctx(self):
        for m in self._hooked_modules:
            try:
                m._mask_hook_ctx = None
            except Exception:
                pass

    def _register_persistent_decoder_mask_hooks(self):
        self._remove_all_decoder_mask_hooks()
        self._hooked_modules = []

        if not (self.use_decoder_mask and self.build_decoder_mask_modules):
            return

        total = 0
        for bi, block in enumerate(self.up):
            adapters = self.decoder_adapters[bi]
            affine_combiners = self.decoder_affine_combiners[bi]
            for name, sub_module in block.named_modules():
                if not isinstance(sub_module, nn.Conv2d):
                    continue
                if not self._should_mask_decoder_conv(name, sub_module):
                    continue
                k = name.replace('.', '_')
                if k not in adapters:
                    continue
                if self.decoder_mask_use_affine and k not in affine_combiners:
                    continue
                sub_module._mask_hook_ctx = None
                hook = sub_module.register_forward_hook(get_decoder_conv_hook_fn(layer_name=name))
                self._mask_hooks.append(hook)
                self._hooked_modules.append(sub_module)
                total += 1

        if self.debug_decoder_mask or self.decoder_mask_build_verbose:
            print(f"[DecoderMaskBuild][PersistentHooks] conv={total}")

    def _register_decoder_conv_ctx(self, module, sample, mask_generator, adapters, affine_combiners, highfreq_cond=None):
        for name, sub_module in module.named_modules():
            if not isinstance(sub_module, nn.Conv2d):
                continue
            if not self._should_mask_decoder_conv(name, sub_module):
                continue
            k = name.replace('.', '_')
            if k not in adapters:
                continue
            affine_combiner = None
            if self.decoder_mask_use_affine:
                if k not in affine_combiners:
                    continue
                affine_combiner = affine_combiners[k]
            sub_module._mask_hook_ctx = {
                "enabled": True,
                "sample": sample,
                "mask_generator": mask_generator,
                "adapter": adapters[k],
                "affine_combiner": affine_combiner,
                "highfreq_cond": highfreq_cond,
            }

    def _resolve_highfreq_cond(self, mask_cond_img=None, mask_cond_highfreq=None, device=None, dtype=None):
        hf = mask_cond_highfreq
        if hf is None and mask_cond_img is not None:
            hf = self.extract_high_frequency(mask_cond_img)
        if hf is None and self._stored_mask_cond_highfreq is not None:
            hf = self._stored_mask_cond_highfreq
        if hf is None and self._stored_mask_cond_images is not None:
            hf = self.extract_high_frequency(self._stored_mask_cond_images)

        if hf is None:
            return None
        if device is not None:
            hf = hf.to(device=device)
        if dtype is not None:
            hf = hf.to(dtype=dtype)
        return hf

    def forward(self, z, mask_cond_img=None, mask_cond_highfreq=None):
        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        use_decoder_mask = (
            self.use_decoder_mask
            and self.build_decoder_mask_modules
            and len(self.decoder_mask_generators) == len(self.up)
            and len(self.decoder_adapters) == len(self.up)
        )

        highfreq_cond = self._resolve_highfreq_cond(
            mask_cond_img=mask_cond_img,
            mask_cond_highfreq=mask_cond_highfreq,
            device=h.device,
            dtype=h.dtype,
        )

        self._clear_decoder_mask_ctx()
        hf_cache = {}

        for i_level in reversed(range(self.num_resolutions)):
            if use_decoder_mask:
                stage_hf = None
                if highfreq_cond is not None:
                    target_hw = tuple(h.shape[-2:])
                    if tuple(highfreq_cond.shape[-2:]) == target_hw:
                        stage_hf = highfreq_cond
                    elif target_hw in hf_cache:
                        stage_hf = hf_cache[target_hw]
                    else:
                        stage_hf = F.interpolate(highfreq_cond, size=target_hw, mode="nearest")
                        hf_cache[target_hw] = stage_hf

                self._register_decoder_conv_ctx(
                    module=self.up[i_level],
                    sample=h,
                    mask_generator=self.decoder_mask_generators[i_level],
                    adapters=self.decoder_adapters[i_level],
                    affine_combiners=self.decoder_affine_combiners[i_level],
                    highfreq_cond=stage_hf,
                )

            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z

