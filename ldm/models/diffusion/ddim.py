import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import (make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor)


class DDIMSampler(object):
    def __init__(self, diffusion, model, schedule="linear", alpha_generator_func=None, set_alpha_scale=None):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=False,
        )
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=False,
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev
            )
        )
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def _blend_with_noised_source(self, img, x0, mask, timesteps):
        if mask is None:
            return img
        if x0 is None:
            raise ValueError("x0 must be provided when mask is used for blended diffusion")
        img_orig = self.diffusion.q_sample(x0, timesteps)
        return img_orig * mask + (1.0 - mask) * img

    def _normalize_blend_start_step(self, blend_start_step, total_steps):
        if blend_start_step is None:
            return total_steps - 1
        blend_start_step = int(blend_start_step)
        if blend_start_step < 0:
            blend_start_step = 0
        if blend_start_step > total_steps - 1:
            blend_start_step = total_steps - 1
        return blend_start_step

    @torch.no_grad()
    def _prepare_sampling_loop(self, shape, input, x0=None, blend_start_step=None):
        b = shape[0]
        total_steps = self.ddim_timesteps.shape[0]
        start_index = self._normalize_blend_start_step(blend_start_step, total_steps)
        indexes = list(range(start_index, -1, -1))

        img = input["x"]
        if img is None:
            if blend_start_step is not None:
                if x0 is None:
                    raise ValueError("x0 must be provided when blend_start_step is used")
                ts_start = torch.full((b,), int(self.ddim_timesteps[start_index]), device=self.device, dtype=torch.long)
                img = self.diffusion.q_sample(x0, ts_start)
            else:
                img = torch.randn(shape, device=self.device)
            input["x"] = img

        return img, indexes, total_steps, start_index

    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None, blend_start_step=None):
        self.make_schedule(ddim_num_steps=S)
        return self.ddim_sampling(
            shape,
            input,
            uc,
            guidance_scale,
            mask=mask,
            x0=x0,
            blend_start_step=blend_start_step,
        )

    @torch.no_grad()
    def ddim_sampling(self, shape, input, uc, guidance_scale=1, mask=None, x0=None, blend_start_step=None):
        img, indexes, total_steps, start_index = self._prepare_sampling_loop(
            shape=shape,
            input=input,
            x0=x0,
            blend_start_step=blend_start_step,
        )
        b = shape[0]

        if self.alpha_generator_func is not None:
            alphas = self.alpha_generator_func(len(indexes))
        else:
            alphas = None

        for i, index in enumerate(indexes):
            step = self.ddim_timesteps[index]

            if alphas is not None:
                self.set_alpha_scale(self.model, alphas[i])
                if alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            input["timesteps"] = torch.full((b,), int(step), device=self.device, dtype=torch.long)

            if mask is not None:
                img = self._blend_with_noised_source(img, x0, mask, input["timesteps"])
                input["x"] = img

            img, pred_x0 = self.p_sample_ddim(input, index=index, uc=uc, guidance_scale=guidance_scale)
            input["x"] = img

        return img

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(self, input, uc=None, guidance_scale=1, mask=None, x0=None, t_start=None):
        assert t_start is not None, "t_start 不能为空，必须指定从哪个 DDIM 步开始解码"
        assert hasattr(self, "ddim_timesteps"), "请先调用 make_schedule() 再使用 decode()"

        img = input["x"]
        b = img.shape[0]

        total_steps = self.ddim_timesteps.shape[0]
        start_index = self._normalize_blend_start_step(t_start, total_steps)
        indexes = list(range(start_index, -1, -1))

        if self.alpha_generator_func is not None:
            alphas = self.alpha_generator_func(len(indexes))
        else:
            alphas = None

        for i, index in enumerate(indexes):
            step = self.ddim_timesteps[index]

            if alphas is not None:
                self.set_alpha_scale(self.model, alphas[i])
                if alphas[i] == 0:
                    self.model.restore_first_conv_from_SD()

            input["timesteps"] = torch.full((b,), int(step), device=self.device, dtype=torch.long)

            if mask is not None:
                img = self._blend_with_noised_source(img, x0, mask, input["timesteps"])

            input["x"] = img
            img, pred_x0 = self.p_sample_ddim(input, index=index, uc=uc, guidance_scale=guidance_scale)

        return img

    @torch.no_grad()
    def p_sample_ddim(self, input, index, uc=None, guidance_scale=1):
        e_t = self.model(input)
        if uc is not None and guidance_scale != 1:
            unconditional_input = dict(
                x=input["x"],
                timesteps=input["timesteps"],
                context=uc,
                inpainting_extra_input=input["inpainting_extra_input"],
                grounding_extra_input=input['grounding_extra_input'],
            )
            e_t_uncond = self.model(unconditional_input)
            e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        b = input["x"].shape[0]
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=self.device)

        pred_x0 = (input["x"] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(input["x"])
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
