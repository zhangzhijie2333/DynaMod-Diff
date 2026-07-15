import torch
import torch.nn as nn
#import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model_weight_mask_第一版_enable_affine开关版_sem版 import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config




class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 scale_factor=1
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self._decoder_ref_images = None
        self._decoder_ref_highfreq = None

    def set_decoder_reference_images(self, images, mode=None, keep_highfreq=True):
        self._decoder_ref_images = images
        if keep_highfreq and hasattr(self.decoder, "extract_high_frequency") and images is not None:
            self._decoder_ref_highfreq = self.decoder.extract_high_frequency(images, mode=mode)
        else:
            self._decoder_ref_highfreq = None

    def clear_decoder_reference_images(self):
        self._decoder_ref_images = None
        self._decoder_ref_highfreq = None
        if hasattr(self.decoder, "clear_mask_reference_images"):
            self.decoder.clear_mask_reference_images()

    def extract_decoder_high_frequency(self, images, mode=None):
        if not hasattr(self.decoder, "extract_high_frequency"):
            raise AttributeError("Current decoder does not implement extract_high_frequency")
        return self.decoder.extract_high_frequency(images, mode=mode)

    def get_decoder_mask_build_stats(self):
        if hasattr(self.decoder, "get_decoder_mask_build_stats"):
            return self.decoder.get_decoder_mask_build_stats()
        return getattr(self.decoder, "decoder_mask_build_stats", None)

    def decoder_affine_enabled(self):
        return bool(getattr(self.decoder, "decoder_enable_affine_combiner", getattr(self.decoder, "decoder_mask_use_affine", False)))

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample() * self.scale_factor

    def decode(self, z, decoder_ref_images=None, decoder_ref_highfreq=None, grounding_extra_input=None):
        z = 1. / self.scale_factor * z
        z = self.post_quant_conv(z)

        if decoder_ref_highfreq is None and decoder_ref_images is None and self._decoder_ref_highfreq is not None:
            decoder_ref_highfreq = self._decoder_ref_highfreq
        elif decoder_ref_highfreq is None and decoder_ref_images is None and self._decoder_ref_images is not None:
            decoder_ref_images = self._decoder_ref_images

        dec = self.decoder(
            z,
            mask_cond_img=decoder_ref_images,
            mask_cond_highfreq=decoder_ref_highfreq,
            grounding_extra_input=grounding_extra_input,
        )
        return dec








