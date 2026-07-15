import torch
from ldm.models.diffusion.ddim_第三代 import DDIMSampler, predict_x0_hat_from_epsilon
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
import random
import time 
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import os 
import shutil
import torchvision
from PIL import Image
import matplotlib.cm as cm
from torchvision.transforms import functional as TF
from torchvision import transforms
from convert_ckpt import add_additional_channels
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize, get_world_size
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, CLIPModel, AutoTokenizer
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
from ldm.modules.attention import BasicTransformerBlock
import re

try:
    import kornia.augmentation as K
except Exception:
    K = None
try:
    from apex import amp
except:
    pass  
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #



class ImageCaptionSaver:
    def __init__(self, base_path, nrow=8, normalize=True, scale_each=True, value_range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.value_range = value_range

    def __call__(self, images, real, masked_real, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        torchvision.utils.save_image( images, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.value_range )
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_real.png')
        torchvision.utils.save_image( real, save_path, nrow=self.nrow)

        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, value_range=self.value_range)

        assert images.shape[0] == len(captions)

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                f.write( cap + '\n' )  
            f.write( '\n' ) 


import os
import torch

def read_official_ckpt(ckpt_path):
    """Read official pretrained SD ckpt and convert into my style.
    Compatible with:
      1) Lightning checkpoint: {"state_dict": {...}}
      2) Plain state_dict: {...}
    """
    print("=" * 80)
    print(f"[CKPT] Loading official checkpoint: {ckpt_path}")
    print(f"[CKPT] Abs path: {os.path.abspath(ckpt_path)}")
    try:
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"[CKPT] File size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"[CKPT] File size: <unknown> ({e})")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
        print("[CKPT] Format: lightning checkpoint (ckpt['state_dict'])")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        print("[CKPT] Format: plain state_dict (no outer 'state_dict')")
    else:
        raise TypeError(f"[CKPT] Unexpected checkpoint type: {type(ckpt)}")

    print(f"[CKPT] Loaded raw keys: {len(state_dict)}")
    print("=" * 80)

    out = {
        "model": {},
        "text_encoder": {},
        "autoencoder": {},
        "unexpected": {},
        "diffusion": {},
    }

    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model"):
            out["model"][k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("cond_stage_model"):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v
        elif k.startswith("first_stage_model"):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v
        else:
            out["diffusion"][k] = v

    print(f"[CKPT] Split summary: "
          f"model={len(out['model'])}, text_encoder={len(out['text_encoder'])}, "
          f"autoencoder={len(out['autoencoder'])}, diffusion={len(out['diffusion'])}, "
          f"unexpected={len(out['unexpected'])}")

    return out


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 





def _is_affine_key(name):
    return ("linear_affine_combiners" in name) or ("conv_affine_combiners" in name)


class DifferentiableImageAugmentations(nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7, use_perspective=False):
        super().__init__()
        self.output_size = int(output_size)
        self.augmentations_number = max(int(augmentations_number), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        if K is not None:
            aug_modules = [
                K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),
            ]
            if bool(use_perspective):
                aug_modules.append(K.RandomPerspective(0.7, p=p))
            self.augmentations = nn.Sequential(*aug_modules)
        else:
            self.augmentations = None

    def forward(self, input_tensor):
        resized_images = self.avg_pool(input_tensor)
        if self.augmentations_number <= 1:
            return resized_images

        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))
        batch_size = input_tensor.shape[0]
        non_augmented_batch = resized_images[:batch_size]
        augmented_source = resized_images[batch_size:]
        if augmented_source.numel() == 0:
            return non_augmented_batch
        if self.augmentations is not None:
            augmented_batch = self.augmentations(augmented_source)
        else:
            augmented_batch = augmented_source
        return torch.cat([non_augmented_batch, augmented_batch], dim=0)


class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")
        self.use_amp = bool(getattr(config, "use_amp", True))
        self.scaler = GradScaler(enabled=self.use_amp)

        self.l_simple_weight = 1
        self.lambda_eps = float(getattr(config, "lambda_eps", 1.0))
        self.lambda_img = float(getattr(config, "lambda_img", 0.0))
        self.lambda_pix = float(getattr(config, "lambda_pix", 1.0))
        self.lambda_perc = float(getattr(config, "lambda_perc", 0.0))
        self.lambda_sem = float(getattr(config, "lambda_sem", 1.0))
        self.lambda_bg = float(getattr(config, "lambda_bg", 1.0))
        self.lambda_boundary = float(getattr(config, "lambda_boundary", 1.0))
        self.use_loss_img = bool(getattr(config, "use_loss_img", True))
        self.use_loss_img_pix = bool(getattr(config, "use_loss_img_pix", True))
        self.use_loss_img_perc = bool(getattr(config, "use_loss_img_perc", True))
        self.use_loss_img_sem = bool(getattr(config, "use_loss_img_sem", True))
        self.use_loss_img_bg = bool(getattr(config, "use_loss_img_bg", False))
        self.use_loss_img_boundary = bool(getattr(config, "use_loss_img_boundary", False))
        self.use_loss_img_t_weight = bool(getattr(config, "use_loss_img_t_weight", True))
        self.image_loss_weight_power = float(getattr(config, "image_loss_weight_power", 1.0))
        self.boundary_kernel_size = int(getattr(config, "boundary_kernel_size", 5))
        self.boundary_dilate_iter = int(getattr(config, "boundary_dilate_iter", 1))
        self._warned_no_sem_for_image_loss = False
        self.decoder_ref_image_path = getattr(config, "decoder_ref_image_path", None)
        self.decoder_ref_image_size = int(getattr(config, "decoder_ref_image_size", 256))
        self.decoder_ref_image_repeat_to_batch = bool(getattr(config, "decoder_ref_image_repeat_to_batch", True))
        self.decoder_train_full = bool(getattr(config, "decoder_train_full", True))
        self.decoder_debug_every = int(getattr(config, "decoder_debug_every", 50))
        self.log_loss_every = int(getattr(config, "log_loss_every", 10))
        self.use_loss_attn = bool(getattr(config, "use_loss_attn", False))
        self.lambda_attn = float(getattr(config, "lambda_attn", 0.0))
        self.attn_loss_normalize = bool(getattr(config, "attn_loss_normalize", True))
        self.attn_supervision_size = int(getattr(config, "attn_supervision_size", 256))
        self.attn_use_native_grounding_mask = bool(getattr(config, "attn_use_native_grounding_mask", True))
        self.attn_supervision_interp = str(getattr(config, "attn_supervision_interp", "bicubic"))
        self.attn_save_panel_only = bool(getattr(config, "attn_save_panel_only", True))
        self.save_attn_heatmap_every = int(getattr(config, "save_attn_heatmap_every", 0))
        self.save_attn_heatmap_layers = bool(getattr(config, "save_attn_heatmap_layers", False))
        self.save_attn_heatmap_max_samples = int(getattr(config, "save_attn_heatmap_max_samples", 1))
        self.use_loss_clip_mask = bool(getattr(config, "use_loss_clip_mask", False))
        self.lambda_clip_mask = float(getattr(config, "lambda_clip_mask", 0.0))
        self.clip_mask_model_path = str(getattr(config, "clip_mask_model_path", "openai/clip-vit-large-patch14"))
        self.clip_mask_aug_num = max(int(getattr(config, "clip_mask_aug_num", 4)), 1)
        self.clip_mask_aug_p = float(getattr(config, "clip_mask_aug_p", 0.7))
        self.clip_mask_use_perspective = bool(getattr(config, "clip_mask_use_perspective", False))
        self.clip_mask_use_colon_suffix = bool(getattr(config, "clip_mask_use_colon_suffix", True))
        self.clip_mask_use_mask = bool(getattr(config, "clip_mask_use_mask", True))
        self.clip_mask_log_every = int(getattr(config, "clip_mask_log_every", 50))
        self.save_first_clip_mask_pred_x0 = bool(getattr(config, "save_first_clip_mask_pred_x0", True))
        self.first_clip_mask_save_iter = int(getattr(config, "first_clip_mask_save_iter", 0))
        self._first_clip_mask_pred_x0_saved = False
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )

        self.attn_heatmap_dir = os.path.join(self.name, "AttentionHeatmaps")
        self.clip_mask_debug_dir = os.path.join(self.name, "ClipMaskDebug")
        if get_rank() == 0 and self.save_attn_heatmap_every > 0:
            os.makedirs(self.attn_heatmap_dir, exist_ok=True)
        if get_rank() == 0 and self.save_first_clip_mask_pred_x0:
            os.makedirs(self.clip_mask_debug_dir, exist_ok=True)


        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        # state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        print(f"[CKPT] Loading official checkpoint: {config.official_ckpt_name}")
        state_dict = read_official_ckpt(config.official_ckpt_name)

        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels > 0 else False

        # load original SD ckpt (with input conv may be modified)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict["model"], strict=False)
        assert unexpected_keys == []
        original_params_names = list(state_dict["model"].keys())  # used for sanity check later

        missing_ae, unexpected_ae = self.autoencoder.load_state_dict(
            state_dict["autoencoder"], strict=False
        )
        print("[autoencoder load] missing:", len(missing_ae))
        print("[autoencoder load] unexpected:", len(unexpected_ae))

        self.text_encoder.load_state_dict(state_dict["text_encoder"], strict=False)
        self.diffusion.load_state_dict(state_dict["diffusion"])

        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)
        self.autoencoder.train()
        self.autoencoder.encoder.eval()
        disable_grads(self.autoencoder.encoder)
        if hasattr(self.autoencoder, "quant_conv"):
            disable_grads(self.autoencoder.quant_conv)
            self.autoencoder.quant_conv.eval()
        self._decoder_ref_image_tensor = self._load_decoder_reference_image(self.decoder_ref_image_path)
        if self._decoder_ref_image_tensor is not None and hasattr(self.autoencoder, "set_decoder_reference_images"):
            self.autoencoder.set_decoder_reference_images(self._decoder_ref_image_tensor)

        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_size = None
        self.clip_normalize = None
        self.clip_augmentations = None
        if self.use_loss_clip_mask and self.lambda_clip_mask > 0:
            self.clip_model = CLIPModel.from_pretrained(self.clip_mask_model_path, local_files_only=True).to(self.device).eval()
            self.clip_tokenizer = AutoTokenizer.from_pretrained(self.clip_mask_model_path, local_files_only=True)
            disable_grads(self.clip_model)
            self.clip_size = int(self.clip_model.config.vision_config.image_size)
            self.clip_normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            self.clip_augmentations = DifferentiableImageAugmentations(
                output_size=self.clip_size,
                augmentations_number=self.clip_mask_aug_num,
                p=self.clip_mask_aug_p,
                use_perspective=self.clip_mask_use_perspective,
            ).to(self.device)
            if get_rank() == 0:
                print(
                    f"[ClipMaskLoss] enabled local_path={self.clip_mask_model_path} clip_size={self.clip_size} "
                    f"aug_num={self.clip_mask_aug_num} aug_p={self.clip_mask_aug_p} "
                    f"use_perspective={self.clip_mask_use_perspective} "
                    f"use_mask={self.clip_mask_use_mask} use_colon_suffix={self.clip_mask_use_colon_suffix}"
                )

        ckpt_param_names = set()

        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")

            missing, unexpected = self.model.load_state_dict(first_stage_ckpt["model"], strict=False)
            self._handle_unexpected_keys(unexpected, stage="load ckpt")
            print("[load ckpt] missing:", len(missing))
            print("[load ckpt] unexpected:", len(unexpected))
            ckpt_param_names = set(first_stage_ckpt["model"].keys())

        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        disable_grads(self.model)

        # -------- model: only train selected modules --------
        model_mask_name_tokens = (
            "linear_mask_generators",
            "linear_adapters",
            "linear_affine_combiners",
            "conv_mask_generators",
            "conv_adapters",
            "conv_affine_combiners",
        )

        for name, p in self.model.named_parameters():
            is_fuser_param = ("transformer_blocks" in name) and ("fuser" in name)
            is_position_net_param = "position_net" in name
            is_downsample_net_param = "downsample_net" in name
            is_input_conv_param = self.input_conv_train and (name == "input_blocks.0.0.weight")
            is_mask_param = any(token in name for token in model_mask_name_tokens)

            should_train = (
                    is_fuser_param
                    or is_position_net_param
                    or is_downsample_net_param
                    or is_input_conv_param
                    or is_mask_param
            )

            if should_train:
                p.requires_grad = True
                params.append(p)
                trainable_names.append(f"model.{name}")
            else:
                p.requires_grad = False
                if not (
                        name in original_params_names
                        or name in ckpt_param_names
                        or is_fuser_param
                        or is_position_net_param
                        or is_downsample_net_param
                        or is_input_conv_param
                        or is_mask_param
                ):
                    print(f"[Warn][new frozen model param] {name}")

            all_params_name.append(f"model.{name}")

        # -------- autoencoder: train ALL decoder params only --------
        self._autoencoder_trainable_names = []

        for name, p in self.autoencoder.named_parameters():

            train_ae = name.startswith("decoder.")

            if train_ae:
                p.requires_grad = True
                params.append(p)
                self._autoencoder_trainable_names.append(f"autoencoder.{name}")
                trainable_names.append(f"autoencoder.{name}")
            else:
                p.requires_grad = False

            all_params_name.append(f"autoencoder.{name}")

        self.opt = torch.optim.AdamW(
            params,
            lr=config.base_learning_rate,
            weight_decay=config.weight_decay,
        )
        count_params(params)
        if get_rank() == 0:
            print(f"[Trainable][model] {len(trainable_names)} params/groups enabled")
            print(f"[Trainable][autoencoder] {len(self._autoencoder_trainable_names)} params/groups enabled")
            for n in self._autoencoder_trainable_names[:40]:
                print(f"  [AE trainable] {n}")
            if len(self._autoencoder_trainable_names) > 40:
                print(f"  ... ({len(self._autoencoder_trainable_names)-40} more)")
            self._print_decoder_mask_build_info()

        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters())
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.total_iters,
            )
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(
                self.opt,
                num_warmup_steps=config.warmup_steps,
            )
        else:
            assert False

        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None
        dataset_train = ConCatDataset(
            config.train_dataset_names,
            config.DATA_ROOT,
            train=True,
            repeats=train_dataset_repeats,
        )
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None
        loader_train = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=(sampler is None),
            num_workers=config.workers,
            pin_memory=True,
            sampler=sampler,
        )
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)

        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
            self._handle_unexpected_keys(unexpected, stage="auto resume")
            if "autoencoder" in checkpoint:
                ae_missing, ae_unexpected = self.autoencoder.load_state_dict(checkpoint["autoencoder"], strict=False)
                print("[auto resume][autoencoder] missing:", len(ae_missing))
                print("[auto resume][autoencoder] unexpected:", len(ae_unexpected))
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()

        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#

        # func return input for grounding tokenizer
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input

        # func return input for grounding downsampler
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:
            self.image_caption_saver = ImageCaptionSaver(self.name)

        if config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                broadcast_buffers=False,
            )

    def _model_wo_wrapper(self):
        return self.model.module if self.config.distributed else self.model

    def _affine_enabled(self):
        model_wo_wrapper = self._model_wo_wrapper()
        return bool(getattr(model_wo_wrapper, "enable_affine_combiner", True))

    def _handle_unexpected_keys(self, unexpected, stage="load"):
        unexpected = list(unexpected)
        if not unexpected:
            return []

        if self._affine_enabled():
            raise AssertionError(f"[{stage}] unexpected keys found: {unexpected[:20]}")

        affine_unexpected = [k for k in unexpected if _is_affine_key(k)]
        remain = [k for k in unexpected if not _is_affine_key(k)]

        if affine_unexpected and get_rank() == 0:
            print(f"[{stage}] ignore affine unexpected keys because enable_affine_combiner=False: {len(affine_unexpected)}")
            for k in affine_unexpected[:20]:
                print(f"  [ignored] {k}")
            if len(affine_unexpected) > 20:
                print(f"  ... ({len(affine_unexpected) - 20} more)")

        if remain:
            raise AssertionError(f"[{stage}] unexpected non-affine keys found: {remain[:20]}")

        return affine_unexpected


    def _autoencoder_wo_wrapper(self):
        return self.autoencoder

    def _load_decoder_reference_image(self, path):
        if path is None or str(path).lower() == "none" or str(path).strip() == "":
            if get_rank() == 0:
                print("[DecoderRef] no reference image path provided; decoder mask will use None")
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"decoder_ref_image_path not found: {path}")
        image = Image.open(path).convert("RGB")
        if self.decoder_ref_image_size > 0:
            image = image.resize((self.decoder_ref_image_size, self.decoder_ref_image_size), resample=Image.BICUBIC)
        tensor = TF.to_tensor(image).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.to(self.device)
        if get_rank() == 0:
            print(f"[DecoderRef] loaded {path}, shape={tuple(tensor.shape)}")
        return tensor

    def _get_decoder_reference_batch(self, batch_size):
        if self._decoder_ref_image_tensor is None:
            return None
        ref = self._decoder_ref_image_tensor
        if self.decoder_ref_image_repeat_to_batch and ref.size(0) == 1 and batch_size > 1:
            ref = ref.expand(batch_size, -1, -1, -1).contiguous()
        return ref

    def _compute_image_loss_weight(self, t):
        denom = max(int(self.diffusion.num_timesteps) - 1, 1)
        w = 1.0 - (t.float() / float(denom))
        w = w.clamp(min=0.0, max=1.0)
        if self.image_loss_weight_power != 1.0:
            w = w.pow(self.image_loss_weight_power)
        return w

    def predict_x0_hat(self, x_t, timesteps, eps_pred):
        return predict_x0_hat_from_epsilon(self.diffusion.alphas_cumprod, x_t, timesteps, eps_pred)

    def _extract_ae_encoder_features(self, x):
        encoder = self._autoencoder_wo_wrapper().encoder
        temb = None
        feats = []
        h = encoder.conv_in(x)
        feats.append(h)
        hs = [h]
        for i_level in range(encoder.num_resolutions):
            for i_block in range(encoder.num_res_blocks):
                h = encoder.down[i_level].block[i_block](hs[-1], temb)
                if len(encoder.down[i_level].attn) > 0:
                    h = encoder.down[i_level].attn[i_block](h)
                hs.append(h)
                feats.append(h)
            if i_level != encoder.num_resolutions - 1:
                h = encoder.down[i_level].downsample(hs[-1])
                hs.append(h)
                feats.append(h)
        h = hs[-1]
        h = encoder.mid.block_1(h, temb)
        feats.append(h)
        h = encoder.mid.attn_1(h)
        feats.append(h)
        h = encoder.mid.block_2(h, temb)
        feats.append(h)
        return feats

    def _per_sample_l1(self, pred, target):
        return (pred - target).abs().flatten(1).mean(dim=1)

    def _per_sample_perc_loss(self, pred, target):
        pred_feats = self._extract_ae_encoder_features(pred)
        with torch.no_grad():
            target_feats = self._extract_ae_encoder_features(target)
        losses = []
        for pf, tf in zip(pred_feats, target_feats):
            losses.append((pf - tf).abs().flatten(1).mean(dim=1))
        return torch.stack(losses, dim=0).mean(dim=0)

    def _extract_sem_binary_mask(self, ref_tensor=None, grounding_extra_input=None, batch=None, resize_to=None, force_dtype=None, force_device=None):
        sem = None
        sem_source_name = None

        if grounding_extra_input is not None:
            sem = grounding_extra_input
            sem_source_name = "grounding_extra_input"
        elif isinstance(batch, dict) and ("sem" in batch):
            sem = batch["sem"]
            sem_source_name = "batch['sem']"

        if sem is None:
            need_sem_mask_loss = self.use_loss_img_sem or self.use_loss_img_bg or self.use_loss_img_boundary or self.use_loss_attn
            if need_sem_mask_loss and (not self._warned_no_sem_for_image_loss) and get_rank() == 0:
                print("[SemMaskLoss] no grounding_extra_input / batch['sem']; sem/bg/boundary/attn losses will be zero.")
                self._warned_no_sem_for_image_loss = True
            return None

        if not torch.is_tensor(sem):
            raise TypeError(f"{sem_source_name} must be a torch.Tensor, got {type(sem)}")

        if sem.dim() == 3:
            sem = sem.unsqueeze(1)
        elif sem.dim() != 4:
            raise ValueError(f"{sem_source_name} must be 3D or 4D, got shape={tuple(sem.shape)}")

        device = force_device if force_device is not None else (ref_tensor.device if ref_tensor is not None else sem.device)
        dtype = force_dtype if force_dtype is not None else (ref_tensor.dtype if ref_tensor is not None else sem.dtype)

        sem = sem.to(device=device)
        if not torch.is_floating_point(sem):
            sem = sem.float()
        sem = sem.to(dtype=dtype)

        if sem.shape[1] == 1:
            sem_mask = (sem > 0.5).to(dtype=dtype)
        else:
            sem_mask = (sem[:, 1:, :, :].amax(dim=1, keepdim=True) > 0.5).to(dtype=dtype)

        if resize_to is not None and tuple(sem_mask.shape[-2:]) != tuple(resize_to):
            sem_mask = torch.nn.functional.interpolate(sem_mask, size=resize_to, mode="nearest")

        return sem_mask.clamp_(0.0, 1.0)

    def _per_sample_masked_l1(self, pred, target, mask):
        diff = (pred - target).abs() * mask
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * float(pred.shape[1])
        return diff.sum(dim=(1, 2, 3)) / denom

    def _build_boundary_band(self, sem_mask):
        kernel_size = max(int(self.boundary_kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        iterations = max(int(self.boundary_dilate_iter), 1)
        padding = kernel_size // 2

        dilated = sem_mask
        eroded = sem_mask
        for _ in range(iterations):
            dilated = torch.nn.functional.max_pool2d(dilated, kernel_size=kernel_size, stride=1, padding=padding)
            eroded = -torch.nn.functional.max_pool2d(-eroded, kernel_size=kernel_size, stride=1, padding=padding)

        boundary = (dilated - eroded) > 0.5
        return boundary.to(dtype=sem_mask.dtype)

    def _compute_image_losses(self, pred, target, t, grounding_extra_input=None, batch=None):
        zero_scalar = pred.new_tensor(0.0)
        one_scalar = pred.new_tensor(1.0)

        if (not self.use_loss_img) or self.lambda_img <= 0:
            return (
                zero_scalar, zero_scalar, zero_scalar, zero_scalar, zero_scalar, zero_scalar,
                one_scalar, zero_scalar, zero_scalar, zero_scalar,
            )

        zero_per_sample = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)

        l1_per_sample = zero_per_sample
        if self.use_loss_img_pix:
            l1_per_sample = self._per_sample_l1(pred, target)

        perc_per_sample = zero_per_sample
        if self.use_loss_img_perc and self.lambda_perc > 0:
            perc_per_sample = self._per_sample_perc_loss(pred, target)

        sem_per_sample = zero_per_sample
        bg_per_sample = zero_per_sample
        boundary_per_sample = zero_per_sample
        sem_mask_mean = zero_per_sample
        bg_mask_mean = zero_per_sample
        boundary_mask_mean = zero_per_sample

        sem_mask = None
        need_sem_mask = (
            (self.use_loss_img_sem and self.lambda_sem > 0)
            or (self.use_loss_img_bg and self.lambda_bg > 0)
            or (self.use_loss_img_boundary and self.lambda_boundary > 0)
        )
        if need_sem_mask:
            sem_mask = self._extract_sem_binary_mask(
                pred,
                grounding_extra_input=grounding_extra_input,
                batch=batch,
                resize_to=tuple(pred.shape[-2:]),
            )

        if sem_mask is not None:
            sem_mask_mean = sem_mask.flatten(1).mean(dim=1)

            if self.use_loss_img_sem and self.lambda_sem > 0:
                sem_per_sample = self._per_sample_masked_l1(pred, target, sem_mask)

            if self.use_loss_img_bg and self.lambda_bg > 0:
                bg_mask = (1.0 - sem_mask).clamp_(0.0, 1.0)
                bg_per_sample = self._per_sample_masked_l1(pred, target, bg_mask)
                bg_mask_mean = bg_mask.flatten(1).mean(dim=1)

            if self.use_loss_img_boundary and self.lambda_boundary > 0:
                boundary_mask = self._build_boundary_band(sem_mask)
                boundary_per_sample = self._per_sample_masked_l1(pred, target, boundary_mask)
                boundary_mask_mean = boundary_mask.flatten(1).mean(dim=1)

        if self.use_loss_img_t_weight:
            w_t = self._compute_image_loss_weight(t)
        else:
            w_t = torch.ones_like(zero_per_sample)

        img_per_sample = zero_per_sample
        if self.use_loss_img_pix:
            img_per_sample = img_per_sample + self.lambda_pix * l1_per_sample
        if self.use_loss_img_perc:
            img_per_sample = img_per_sample + self.lambda_perc * perc_per_sample
        if self.use_loss_img_sem:
            img_per_sample = img_per_sample + self.lambda_sem * sem_per_sample
        if self.use_loss_img_bg:
            img_per_sample = img_per_sample + self.lambda_bg * bg_per_sample
        if self.use_loss_img_boundary:
            img_per_sample = img_per_sample + self.lambda_boundary * boundary_per_sample

        weighted_img = (w_t * img_per_sample).mean()
        total_img_loss = self.lambda_img * weighted_img
        return (
            total_img_loss,
            l1_per_sample.mean(),
            perc_per_sample.mean(),
            sem_per_sample.mean(),
            bg_per_sample.mean(),
            boundary_per_sample.mean(),
            w_t.mean(),
            sem_mask_mean.mean(),
            bg_mask_mean.mean(),
            boundary_mask_mean.mean(),
        )

    def _extract_prompt_suffix_list(self, captions):
        texts = []
        for caption in captions:
            text = str(caption).strip()
            if self.clip_mask_use_colon_suffix and (":" in text):
                suffix = text.split(":", 1)[1].strip()
                if suffix != "":
                    text = suffix
            texts.append(text)
        return texts

    def _encode_clip_text(self, texts):
        if self.clip_model is None or self.clip_tokenizer is None:
            raise RuntimeError("CLIP model/tokenizer is not initialized for clip mask loss")
        tokenized = self.clip_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**tokenized).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return text_features

    def _compute_clip_mask_loss(self, pred, batch, grounding_extra_input=None):
        zero_scalar = pred.new_tensor(0.0)
        zero_cov = pred.new_tensor(0.0)
        if (not self.use_loss_clip_mask) or self.lambda_clip_mask <= 0:
            return zero_scalar, zero_scalar, zero_cov
        if self.clip_model is None or self.clip_augmentations is None or self.clip_normalize is None:
            return zero_scalar, zero_scalar, zero_cov
        if not isinstance(batch, dict) or ("caption" not in batch):
            return zero_scalar, zero_scalar, zero_cov

        sem_mask = self._extract_sem_binary_mask(
            pred,
            grounding_extra_input=grounding_extra_input,
            batch=batch,
            resize_to=tuple(pred.shape[-2:]),
            force_dtype=pred.dtype,
            force_device=pred.device,
        )
        if sem_mask is None:
            return zero_scalar, zero_scalar, zero_cov

        self._save_first_clip_mask_pred_x0(pred, sem_mask, batch.get("caption", None))

        clip_input = pred
        if self.clip_mask_use_mask:
            clip_input = clip_input * sem_mask

        augmented_input = self.clip_augmentations(clip_input.float()).add(1.0).div(2.0).clamp(0.0, 1.0)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.get_image_features(pixel_values=clip_in).float()
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        texts = self._extract_prompt_suffix_list(batch["caption"])
        text_embeds = self._encode_clip_text(texts)
        text_embeds = text_embeds.repeat(self.clip_mask_aug_num, 1)

        dists = 1.0 - (image_embeds * text_embeds).sum(dim=-1)
        batch_size = pred.shape[0]
        per_sample = []
        for i in range(batch_size):
            per_sample.append(dists[i::batch_size].mean())
        per_sample = torch.stack(per_sample, dim=0)
        raw_loss = per_sample.mean()
        weighted_loss = self.lambda_clip_mask * raw_loss
        sem_cov = sem_mask.flatten(1).mean(dim=1).mean()
        return weighted_loss, raw_loss, sem_cov

    @torch.no_grad()
    def _save_first_clip_mask_pred_x0(self, pred, sem_mask, captions):
        if get_rank() != 0:
            return
        if self._first_clip_mask_pred_x0_saved:
            return
        if not self.save_first_clip_mask_pred_x0:
            return
        if int(self.iter_idx) != int(self.first_clip_mask_save_iter):
            return
        if pred is None or sem_mask is None or pred.shape[0] == 0:
            return

        os.makedirs(self.clip_mask_debug_dir, exist_ok=True)
        sample = pred[:1].detach().float().cpu().clamp(-1.0, 1.0)
        mask = sem_mask[:1].detach().float().cpu().clamp(0.0, 1.0)
        masked = (sample * mask).clamp(-1.0, 1.0)
        overlay = ((sample + 1.0) * 0.5 * 0.6 + mask.repeat(1, 3, 1, 1) * 0.4).clamp(0.0, 1.0)
        sample_vis = (sample + 1.0) * 0.5
        masked_vis = (masked + 1.0) * 0.5
        panel = torch.cat([sample_vis, mask.repeat(1, 3, 1, 1), masked_vis, overlay], dim=0)
        base = os.path.join(self.clip_mask_debug_dir, f"iter_{int(self.iter_idx):07d}_sample_00")
        torchvision.utils.save_image(sample_vis, base + "_pred_x0.png")
        torchvision.utils.save_image(mask.repeat(1, 3, 1, 1), base + "_mask.png")
        torchvision.utils.save_image(masked_vis, base + "_pred_x0_masked.png")
        torchvision.utils.save_image(panel, base + "_panel.png", nrow=4)
        with open(base + "_caption.txt", "w", encoding="utf-8") as f:
            if isinstance(captions, (list, tuple)) and len(captions) > 0:
                f.write(str(captions[0]))
            else:
                f.write("")
        self._first_clip_mask_pred_x0_saved = True
        print(f"[ClipMaskLoss] saved first masked pred_x0 panel to {base}_panel.png")

    def _normalize_attention_map(self, attn_map):
        if attn_map.numel() == 0:
            return attn_map
        flat = attn_map.flatten(1)
        vmin = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        vmax = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        return (attn_map - vmin) / (vmax - vmin + 1e-6)

    def _compute_attention_loss(self, latent_ref, grounding_extra_input=None, batch=None, need_loss=True, need_visual=False):
        zero_scalar = latent_ref.new_tensor(0.0)
        payload = None

        model_wo_wrapper = self._model_wo_wrapper()
        if not hasattr(model_wo_wrapper, "get_collected_gated_sa_maps"):
            return zero_scalar, zero_scalar, 0, zero_scalar, payload

        attn_records = model_wo_wrapper.get_collected_gated_sa_maps()
        if not attn_records:
            return zero_scalar, zero_scalar, 0, zero_scalar, payload

        target_mask = self._extract_sem_binary_mask(
            ref_tensor=None,
            grounding_extra_input=grounding_extra_input,
            batch=batch,
            resize_to=None,
            force_dtype=latent_ref.dtype,
            force_device=latent_ref.device,
        )
        if target_mask is None:
            return zero_scalar, zero_scalar, 0, zero_scalar, payload

        native_target_size = tuple(target_mask.shape[-2:])
        if self.attn_use_native_grounding_mask:
            target_size = native_target_size
        elif self.attn_supervision_size and self.attn_supervision_size > 0:
            target_size = (self.attn_supervision_size, self.attn_supervision_size)
        else:
            target_size = native_target_size

        if tuple(target_mask.shape[-2:]) != tuple(target_size):
            target_mask = torch.nn.functional.interpolate(target_mask, size=target_size, mode="nearest")

        interp_mode = str(self.attn_supervision_interp).lower()
        if interp_mode not in {"nearest", "bilinear", "bicubic"}:
            interp_mode = "bicubic"
        align_corners = False if interp_mode in {"bilinear", "bicubic"} else None

        layer_maps = []
        layer_names = []
        for record in attn_records:
            attn = record.get("attn", None)
            if attn is None or attn.dim() != 4 or attn.numel() == 0:
                continue

            heat = attn.mean(dim=-1).mean(dim=1)  # [B, N_visual]
            visual_hw = record.get("visual_hw", (None, None))
            if not isinstance(visual_hw, (tuple, list)) or len(visual_hw) != 2:
                visual_hw = (None, None)
            h_map, w_map = visual_hw
            if h_map is None or w_map is None:
                n_visual = heat.shape[-1]
                side = int(math.sqrt(n_visual))
                if side * side != n_visual:
                    continue
                h_map, w_map = side, side

            heat = heat.view(heat.shape[0], 1, h_map, w_map)
            if interp_mode == "nearest":
                heat = torch.nn.functional.interpolate(heat, size=target_size, mode=interp_mode)
            else:
                heat = torch.nn.functional.interpolate(
                    heat,
                    size=target_size,
                    mode=interp_mode,
                    align_corners=align_corners,
                )
            layer_maps.append(heat)
            layer_names.append(record.get("name", f"layer_{len(layer_names)}"))

        if not layer_maps:
            return zero_scalar, zero_scalar, 0, zero_scalar, payload

        attn_map = torch.stack(layer_maps, dim=0).mean(dim=0)
        if self.attn_loss_normalize:
            attn_map = self._normalize_attention_map(attn_map)

        raw_loss = torch.nn.functional.mse_loss(attn_map, target_mask)
        weighted_loss = self.lambda_attn * raw_loss if need_loss and self.use_loss_attn and self.lambda_attn > 0 else zero_scalar

        if need_visual:
            payload = {
                "attn_map": attn_map.detach(),
                "target_mask": target_mask.detach(),
                "layer_maps": [m.detach() for m in layer_maps],
                "layer_names": list(layer_names),
            }

        return weighted_loss, raw_loss, len(layer_maps), attn_map.mean(), payload

    def _should_save_attention_heatmaps(self):
        return self.save_attn_heatmap_every > 0 and get_rank() == 0 and (self.iter_idx % self.save_attn_heatmap_every == 0)

    def _colorize_heatmap_tensor(self, heatmap_2d):
        heat_np = heatmap_2d.detach().float().clamp(0.0, 1.0).cpu().numpy()
        colored = cm.get_cmap("jet")(heat_np)[..., :3]
        return torch.from_numpy(colored).permute(2, 0, 1).float()

    def _sanitize_layer_name(self, name):
        safe = []
        for ch in str(name):
            if ch.isalnum() or ch in ['.', '-', '_']:
                safe.append(ch)
            else:
                safe.append('_')
        return ''.join(safe)

    @torch.no_grad()
    def _save_attention_heatmaps(self, batch, payload):
        if payload is None or get_rank() != 0:
            return
        attn_map = payload.get("attn_map", None)
        target_mask = payload.get("target_mask", None)
        if attn_map is None or target_mask is None:
            return

        os.makedirs(self.attn_heatmap_dir, exist_ok=True)
        image = batch.get("image", None)
        if image is None or not torch.is_tensor(image):
            return

        max_samples = max(1, min(self.save_attn_heatmap_max_samples, image.shape[0], attn_map.shape[0], target_mask.shape[0]))
        image_vis = (image[:max_samples].detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        save_hw = tuple(image_vis.shape[-2:])

        attn_up = torch.nn.functional.interpolate(
            attn_map[:max_samples].detach().float().cpu(),
            size=save_hw,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        mask_up = torch.nn.functional.interpolate(
            target_mask[:max_samples].detach().float().cpu(),
            size=save_hw,
            mode="nearest",
        ).clamp(0.0, 1.0)

        for si in range(max_samples):
            base = f"iter_{self.iter_idx:07d}_sample_{si:02d}"
            attn_rgb = self._colorize_heatmap_tensor(attn_up[si, 0])
            mask_rgb = mask_up[si].repeat(3, 1, 1)
            overlay = (0.6 * image_vis[si] + 0.4 * attn_rgb).clamp(0.0, 1.0)
            panel = torch.stack([image_vis[si], attn_rgb, mask_rgb, overlay], dim=0)
            torchvision.utils.save_image(panel, os.path.join(self.attn_heatmap_dir, base + "_panel.png"), nrow=4)

    def _print_decoder_mask_build_info(self):
        ae = self._autoencoder_wo_wrapper()
        decoder = getattr(ae, "decoder", None)
        if decoder is None:
            return
        if get_rank() != 0:
            return
        print("[DecoderMaskBuild][TrainerSummary]")
        print(f"  use_decoder_mask={getattr(decoder, 'use_decoder_mask', None)}")
        print(f"  build_decoder_mask_modules={getattr(decoder, 'build_decoder_mask_modules', None)}")
        print(f"  use_highfreq={getattr(decoder, 'decoder_mask_use_highfreq_cond', None)}")
        print(f"  highfreq_in_dim={getattr(decoder, 'decoder_mask_highfreq_in_dim', None)}")
        print(f"  use_sem={getattr(decoder, 'decoder_mask_use_sem_cond', None)}")
        print(f"  sem_in_dim={getattr(decoder, 'decoder_mask_sem_in_dim', None)}")
        print(f"  use_affine={getattr(decoder, 'decoder_enable_affine_combiner', getattr(decoder, 'decoder_mask_use_affine', None))}")
        print(f"  build_verbose={getattr(decoder, 'decoder_mask_build_verbose', None)}")
        mg = getattr(decoder, "decoder_mask_generators", None)
        ad = getattr(decoder, "decoder_adapters", None)
        af = getattr(decoder, "decoder_affine_combiners", None)
        if mg is not None:
            print(f"  decoder_mask_generators={len(mg)}")
        if ad is not None:
            total_ad = 0
            for bi, item in enumerate(ad):
                n = len(item) if isinstance(item, nn.ModuleDict) else 0
                total_ad += n
                print(f"  decoder_adapters[{bi}]={n}")
            print(f"  decoder_adapters_total={total_ad}")
        if af is not None:
            total_af = 0
            for bi, item in enumerate(af):
                n = len(item) if isinstance(item, nn.ModuleDict) else 0
                total_af += n
                print(f"  decoder_affine_combiners[{bi}]={n}")
            print(f"  decoder_affine_total={total_af}")
        build_stats = None
        if hasattr(decoder, 'get_decoder_mask_build_stats'):
            build_stats = decoder.get_decoder_mask_build_stats()
        else:
            build_stats = getattr(decoder, 'decoder_mask_build_stats', None)
        if build_stats:
            print(f"  decoder_mask_build_total_hooked={getattr(decoder, 'decoder_mask_build_total_hooked', None)}")
            if getattr(decoder, 'decoder_mask_build_verbose', False):
                for s in build_stats:
                    if s.get('in_c', None) is None:
                        print(
                            f"  [DecoderMaskBuild][{int(s['block_index']):02d}] <empty block> | " 
                            f"Linear(total=0, skipped=0, hooked=0) mg_params=0 ad_params=0 affine_params=0 | " 
                            f"Conv(total=0, skipped=0, hooked=0) mg_params=0 ad_params=0 affine_params=0"
                        )
                        continue
                    print(
                        f"  [DecoderMaskBuild][{int(s['block_index']):02d}] in_c={s['in_c']} out_c={s['out_c']} factor={s['factor']} | "
                        f"Linear(total={s['linear_total']}, skipped={s['linear_skipped']}, hooked={s['linear_hooked']}) "
                        f"mg_params={s['linear_mg_params']} ad_params={s['linear_ad_params']} affine_params={s['linear_affine_params']} | "
                        f"Conv(total={s['conv_total']}, skipped={s['conv_skipped']}, hooked={s['conv_hooked']}) "
                        f"mg_params={s['conv_mg_params']} ad_params={s['conv_ad_params']} affine_params={s['conv_affine_params']}"
                    )

    def _debug_check_decoder_mask_grads(self):
        ae = self._autoencoder_wo_wrapper()
        decoder = getattr(ae, "decoder", None)
        if decoder is None:
            print("  [DecoderGrad] decoder=<NOT FOUND>")
            return
        watched = [
            ("decoder_mask_generators", getattr(decoder, "decoder_mask_generators", None)),
            ("decoder_adapters", getattr(decoder, "decoder_adapters", None)),
        ]
        if bool(getattr(decoder, 'decoder_enable_affine_combiner', getattr(decoder, 'decoder_mask_use_affine', True))):
            watched.append(("decoder_affine_combiners", getattr(decoder, "decoder_affine_combiners", None)))
        print(f"[DecoderGradCheck][iter={self.iter_idx}]")
        for group_name, group in watched:
            if group is None:
                print(f"  {group_name}: <NOT FOUND>")
                continue
            param_cnt = 0
            grad_cnt = 0
            grad_abs_mean_sum = 0.0
            grad_abs_max = 0.0
            for p in group.parameters():
                param_cnt += 1
                if p.grad is not None:
                    grad_cnt += 1
                    gmean = p.grad.abs().mean().item()
                    gmax = p.grad.abs().max().item()
                    grad_abs_mean_sum += gmean
                    grad_abs_max = max(grad_abs_max, gmax)
            avg_grad_mean = grad_abs_mean_sum / max(grad_cnt, 1)
            print(
                f"  {group_name}: param_cnt={param_cnt}, grad_not_none={grad_cnt}, "
                f"grad_abs_mean={avg_grad_mean:.8e}, grad_abs_max={grad_abs_max:.8e}"
            )
        if not bool(getattr(decoder, 'decoder_enable_affine_combiner', getattr(decoder, 'decoder_mask_use_affine', True))):
            print("  decoder_affine_combiners: <DISABLED by decoder_enable_affine_combiner=False>")


    @torch.no_grad()
    def get_input(self, batch):

        z = self._autoencoder_wo_wrapper().encode( batch["image"] )

        context = self.text_encoder.encode( batch["caption"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
            masked_z = z*inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, t, context, inpainting_extra_input, grounding_extra_input 


    def run_one_step(self, batch):
        (x_start, t,
         context, inpainting_extra_input, grounding_extra_input) = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)

        model_wo_wrapper = self._model_wo_wrapper()
        should_collect_attn = (self.use_loss_attn and self.lambda_attn > 0) or self._should_save_attention_heatmaps()
        if hasattr(model_wo_wrapper, "set_attention_map_collection"):
            model_wo_wrapper.set_attention_map_collection(should_collect_attn)
            if should_collect_attn and hasattr(model_wo_wrapper, "clear_attention_map_cache"):
                model_wo_wrapper.clear_attention_map_cache()

        model_output = self.model(input)

        loss_eps = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight
        z0_hat = self.predict_x0_hat(x_noisy, t, model_output)

        decoder_ref_images = self._get_decoder_reference_batch(batch_size=x_start.shape[0])
        decoded_x0 = self._autoencoder_wo_wrapper().decode(
            z0_hat,
            decoder_ref_images=decoder_ref_images,
            grounding_extra_input=grounding_extra_input,
        )

        loss_img, loss_pix, loss_perc, loss_sem, loss_bg, loss_boundary, mean_wt, sem_cov, bg_cov, boundary_cov = self._compute_image_losses(
            decoded_x0,
            batch["image"],
            t,
            grounding_extra_input=grounding_extra_input,
            batch=batch,
        )
        loss_clip_mask, loss_clip_mask_raw, clip_mask_cov = self._compute_clip_mask_loss(
            decoded_x0,
            batch=batch,
            grounding_extra_input=grounding_extra_input,
        )
        need_save_attn = self._should_save_attention_heatmaps()
        loss_attn, loss_attn_raw, attn_layers, attn_mean, attn_payload = self._compute_attention_loss(
            x_start,
            grounding_extra_input=grounding_extra_input,
            batch=batch,
            need_loss=(self.use_loss_attn and self.lambda_attn > 0),
            need_visual=need_save_attn,
        )
        if need_save_attn:
            self._save_attention_heatmaps(batch, attn_payload)
        loss = self.lambda_eps * loss_eps + loss_img + loss_attn + loss_clip_mask

        self.loss_dict = {
            "loss": loss.item(),
            "loss_total": loss.item(),
            "loss_eps": loss_eps.item(),
            "loss_img": loss_img.item(),
            "loss_attn": loss_attn.item(),
            "loss_attn_raw": loss_attn_raw.item(),
            "loss_attn_layers": float(attn_layers),
            "loss_attn_map_mean": attn_mean.item() if torch.is_tensor(attn_mean) else float(attn_mean),
            "loss_clip_mask": loss_clip_mask.item(),
            "loss_clip_mask_raw": loss_clip_mask_raw.item(),
            "loss_clip_mask_cov": clip_mask_cov.item() if torch.is_tensor(clip_mask_cov) else float(clip_mask_cov),
            "loss_img_pix": loss_pix.item(),
            "loss_img_perc": loss_perc.item(),
            "loss_img_sem": loss_sem.item(),
            "loss_img_bg": loss_bg.item(),
            "loss_img_boundary": loss_boundary.item(),
            "loss_img_sem_cov": sem_cov.item(),
            "loss_img_bg_cov": bg_cov.item(),
            "loss_img_boundary_cov": boundary_cov.item(),
            "loss_img_wt": mean_wt.item(),
            "x0_hat_abs_mean": z0_hat.detach().abs().mean().item(),
        }

        return loss
#########################################################################
    def _debug_iter_affine_modules(self, kinds=("conv", "linear")):
        if not self._affine_enabled():
            return

        model_wo_wrapper = self._model_wo_wrapper()

        for kind in kinds:
            attr_name = f"{kind}_affine_combiners"
            group = getattr(model_wo_wrapper, attr_name, None)
            if group is None:
                continue

            for bi, d in enumerate(group):
                if not hasattr(d, "items"):
                    continue
                for layer_name, mod in d.items():
                    item_name = f"{attr_name}[{bi}]['{layer_name}']"
                    yield kind, item_name, mod

    def _debug_check_mask_grads(self, kinds=("conv", "linear")):
        model_wo_wrapper = self._model_wo_wrapper()

        watched = []
        for kind in kinds:
            watched.extend([
                (f"{kind}_mask_generators", getattr(model_wo_wrapper, f"{kind}_mask_generators", None)),
                (f"{kind}_adapters", getattr(model_wo_wrapper, f"{kind}_adapters", None)),
            ])
            if self._affine_enabled():
                watched.append(
                    (f"{kind}_affine_combiners", getattr(model_wo_wrapper, f"{kind}_affine_combiners", None))
                )

        print(f"\n[GradCheck][iter={self.iter_idx}]")
        for group_name, group in watched:
            if group is None:
                print(f"  {group_name}: <NOT FOUND>")
                continue

            param_cnt = 0
            grad_cnt = 0
            grad_abs_mean_sum = 0.0
            grad_abs_max = 0.0

            for p in group.parameters():
                param_cnt += 1
                if p.grad is not None:
                    grad_cnt += 1
                    gmean = p.grad.abs().mean().item()
                    gmax = p.grad.abs().max().item()
                    grad_abs_mean_sum += gmean
                    grad_abs_max = max(grad_abs_max, gmax)

            avg_grad_mean = grad_abs_mean_sum / max(grad_cnt, 1)
            print(
                f"  {group_name}: "
                f"param_cnt={param_cnt}, grad_not_none={grad_cnt}, "
                f"grad_abs_mean={avg_grad_mean:.8e}, grad_abs_max={grad_abs_max:.8e}"
            )

        if not self._affine_enabled():
            print("  affine_combiners: <DISABLED by enable_affine_combiner=False>")

    def _debug_list_active_affines(self, kinds=("conv", "linear"), max_print=200, eps=1e-12, name_filter=None):
        if not self._affine_enabled():
            print("[ActiveAffine] skipped because enable_affine_combiner=False")
            return

        active_nonzero = []
        active_zero = []
        inactive_none = []

        for kind, item_name, mod in self._debug_iter_affine_modules(kinds=kinds):
            if (name_filter is not None) and (name_filter not in item_name):
                continue

            if not all(hasattr(mod, x) for x in ["k0", "k1", "k2"]):
                continue

            g0 = mod.k0.grad
            g1 = mod.k1.grad
            g2 = mod.k2.grad

            g0v = None if g0 is None else g0.abs().mean().item()
            g1v = None if g1 is None else g1.abs().mean().item()
            g2v = None if g2 is None else g2.abs().mean().item()

            all_none = (g0 is None) and (g1 is None) and (g2 is None)
            any_nonzero = (
                (g0v is not None and g0v > eps)
                or (g1v is not None and g1v > eps)
                or (g2v is not None and g2v > eps)
            )

            record = (kind, item_name, g0v, g1v, g2v)

            if all_none:
                inactive_none.append(record)
            elif any_nonzero:
                active_nonzero.append(record)
            else:
                active_zero.append(record)

        print(
            f"[ActiveAffine] filter={name_filter}, "
            f"nonzero={len(active_nonzero)}, "
            f"zero={len(active_zero)}, "
            f"none={len(inactive_none)}"
        )

        print("[ActiveAffine] nonzero layers:")
        for kind, name, k0g, k1g, k2g in active_nonzero[:max_print]:
            print(
                f"  [{kind}] {name}: "
                f"k0_grad={k0g}, "
                f"k1_grad={k1g}, "
                f"k2_grad={k2g}"
            )

        print("[ActiveAffine] zero-grad layers:")
        for kind, name, k0g, k1g, k2g in active_zero[:max_print]:
            print(
                f"  [{kind}] {name}: "
                f"k0_grad={k0g}, "
                f"k1_grad={k1g}, "
                f"k2_grad={k2g}"
            )

        print("[ActiveAffine] none-grad layers(sample):")
        for kind, name, _, _, _ in inactive_none[:30]:
            print(f"  [{kind}] {name}")

    def _debug_probe_affine_triplet(self, kinds=("conv", "linear"), name_filter=None):
        if not self._affine_enabled():
            return None

        for kind, item_name, mod in self._debug_iter_affine_modules(kinds=kinds):
            if name_filter is not None and name_filter not in item_name:
                continue

            if all(hasattr(mod, x) for x in ["k0", "k1", "k2"]):
                return kind, item_name, mod

        return None

    def start_training(self):

        iterator = tqdm(
            range(self.starting_iter, self.config.total_iters),
            desc='Training progress',
            disable=get_rank() != 0
        )
        self.model.train()
        self.autoencoder.train()
        self.autoencoder.encoder.eval()

        for iter_idx in iterator:  # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            loss.backward()

            # ===== 反传后做调试检查（conv + linear 统一）=====
            probe_triplet = None
            probe_before = None

            if get_rank() == 0 and (iter_idx % self.decoder_debug_every == 0):
                print(f"\n[Iter {iter_idx}] after backward")

                # 1) 整体检查 mask 三大模块梯度情况（conv + linear）
                self._debug_check_mask_grads(kinds=("conv", "linear"))
                self._debug_check_decoder_mask_grads()

                # 2) 打印所有 affine combiner 中哪些层有梯度
                #    若只看某个层，可改成：
                #    self._debug_list_active_affines(
                #        kinds=("conv", "linear"),
                #        max_print=200,
                #        name_filter="0_in_layers_2"
                #    )
                self._debug_list_active_affines(
                    kinds=("conv", "linear"),
                    max_print=200
                )

                # 3) 抓一个 probe（优先 conv，再 linear；你也可以改顺序）
                probe_triplet = self._debug_probe_affine_triplet(
                    kinds=("conv", "linear")
                )

                if probe_triplet is None:
                    print(f"[AffineProbeGrad][iter={iter_idx}] probe=<NOT FOUND>")
                else:
                    probe_kind, probe_name, probe_mod = probe_triplet

                    def _grad_str(p):
                        if p.grad is None:
                            return "None"
                        return f"{p.grad.abs().mean().item():.8e}"

                    print(
                        f"[AffineProbeGrad][iter={iter_idx}] [{probe_kind}] {probe_name}: "
                        f"k0_grad={_grad_str(probe_mod.k0)}, "
                        f"k1_grad={_grad_str(probe_mod.k1)}, "
                        f"k2_grad={_grad_str(probe_mod.k2)}"
                    )

                    # 在 step 前保存旧值
                    probe_before = {
                        "k0": probe_mod.k0.detach().clone(),
                        "k1": probe_mod.k1.detach().clone(),
                        "k2": probe_mod.k2.detach().clone(),
                    }

            # ===== 真正更新参数 =====
            self.opt.step()

            # ===== step 后打印 probe 的参数变化 =====
            if get_rank() == 0 and (iter_idx % self.decoder_debug_every == 0):
                if probe_triplet is None:
                    print(f"[AffineProbeDelta][iter={iter_idx}] probe=<NOT FOUND>")
                else:
                    probe_kind, probe_name, probe_mod = probe_triplet

                    for key in ["k0", "k1", "k2"]:
                        before = probe_before[key]
                        after = getattr(probe_mod, key).detach()

                        delta_mean = (after - before).abs().mean().item()
                        delta_max = (after - before).abs().max().item()

                        print(
                            f"[AffineProbeDelta][iter={iter_idx}] [{probe_kind}] {probe_name}.{key}: "
                            f"mean_abs_delta={delta_mean:.8e}, "
                            f"max_abs_delta={delta_max:.8e}, "
                            f"before_mean={before.float().mean().item():.8e}, "
                            f"after_mean={after.float().mean().item():.8e}"
                        )

            self.scheduler.step()

            if get_rank() == 0:
                iterator.set_postfix({
                    "loss": f"{self.loss_dict.get('loss_total', 0.0):.4f}",
                    "eps": f"{self.loss_dict.get('loss_eps', 0.0):.4f}",
                    "img": f"{self.loss_dict.get('loss_img', 0.0):.4f}",
                    "attn": f"{self.loss_dict.get('loss_attn', 0.0):.4f}",
                    "clip": f"{self.loss_dict.get('loss_clip_mask', 0.0):.4f}",
                })
                if iter_idx % self.log_loss_every == 0:
                    print(
                        f"[Loss][iter={iter_idx}] total={self.loss_dict.get('loss_total', 0.0):.6f}, "
                        f"eps={self.loss_dict.get('loss_eps', 0.0):.6f}, "
                        f"img={self.loss_dict.get('loss_img', 0.0):.6f}, "
                        f"attn={self.loss_dict.get('loss_attn', 0.0):.6f}, "
                        f"clip={self.loss_dict.get('loss_clip_mask', 0.0):.6f}, "
                        f"clip_raw={self.loss_dict.get('loss_clip_mask_raw', 0.0):.6f}, "
                        f"attn_raw={self.loss_dict.get('loss_attn_raw', 0.0):.6f}, "
                        f"attn_layers={self.loss_dict.get('loss_attn_layers', 0.0):.0f}, "
                        f"pix={self.loss_dict.get('loss_img_pix', 0.0):.6f}, "
                        f"perc={self.loss_dict.get('loss_img_perc', 0.0):.6f}, "
                        f"sem={self.loss_dict.get('loss_img_sem', 0.0):.6f}, "
                        f"bg={self.loss_dict.get('loss_img_bg', 0.0):.6f}, "
                        f"boundary={self.loss_dict.get('loss_img_boundary', 0.0):.6f}, "
                        f"sem_cov={self.loss_dict.get('loss_img_sem_cov', 0.0):.6f}, "
                        f"bg_cov={self.loss_dict.get('loss_img_bg_cov', 0.0):.6f}, "
                        f"boundary_cov={self.loss_dict.get('loss_img_boundary_cov', 0.0):.6f}, "
                        f"clip_cov={self.loss_dict.get('loss_clip_mask_cov', 0.0):.6f}, "
                        f"w={self.loss_dict.get('loss_img_wt', 0.0):.6f}"
                    )

            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)

            if get_rank() == 0:
                if iter_idx % 10 == 0:
                    self.log_loss()

                if (iter_idx == 0) or (iter_idx % self.config.save_every_iters == 0) or (
                        iter_idx == self.config.total_iters - 1):
                    self.save_ckpt_and_result()

            synchronize()

        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if hasattr(model_wo_wrapper, "set_attention_map_collection"):
            model_wo_wrapper.set_attention_map_collection(False)
            if hasattr(model_wo_wrapper, "clear_attention_map_cache"):
                model_wo_wrapper.clear_attention_map_cache()

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)

            
            if "boxes" in batch:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image"][i], "boxes":batch["boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)
                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            else:
                # keypoint case 
                real_images_with_box_drawing = batch["image"]*0.5 + 0.5 
                
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            # extra input for inpainting 
            inpainting_extra_input = None
            if self.config.inpaint_mode:
                z = self._autoencoder_wo_wrapper().encode( batch["image"] )
                inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
                masked_z = z*inpainting_mask
                inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)
            
            grounding_extra_input = None
            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            input = dict( x=None, 
                          timesteps=None, 
                          context=context, 
                          inpainting_extra_input=inpainting_extra_input,
                          grounding_extra_input=grounding_extra_input,
                          grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=3)
            
            autoencoder_wo_wrapper = self._autoencoder_wo_wrapper()
            decoder_ref_images = self._get_decoder_reference_batch(batch_here)
            samples = autoencoder_wo_wrapper.decode(
                samples,
                decoder_ref_images=decoder_ref_images,
                grounding_extra_input=grounding_extra_input,
            ).cpu()
            samples = torch.clamp(samples, min=-1, max=1)

            masked_real_image =  batch["image"]*torch.nn.functional.interpolate(inpainting_mask, size=(512, 512)) if self.config.inpaint_mode else None
            self.image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, batch["caption"], iter_name)

        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    text_encoder = self.text_encoder.state_dict(),
                    autoencoder = self._autoencoder_wo_wrapper().state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )


