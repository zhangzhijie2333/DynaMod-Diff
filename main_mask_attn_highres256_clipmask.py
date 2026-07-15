import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
import random
from trainer import Trainer
from distributed import synchronize
import os 
import torch.multiprocessing as multiprocessing


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="DATA/Mvtec_per-object/train/carpet", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")

    parser.add_argument("--name", type=str,  default="mvtec-carpet", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str,  default="configs/configs.yaml", help="paths to base configs.")

    parser.add_argument("--base_learning_rate", type=float,  default=1e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=1, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="sd-v1-4.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x:x if type(x) == str and x.lower() != "none" else None,  default="",
        help=("If given, then it，will start training from this ckpt"
              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py)"
              "It must be given if inpaint_mode is true")
    )
    
    parser.add_argument('--inpaint_mode', default=False, type=lambda x:x.lower() == "true", help="Train a GLIGEN model in inpaitning setting")
    parser.add_argument('--randomize_fg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance that fg mask will not be a box but a random mask. See code for details")
    parser.add_argument('--random_add_bg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance add arbitrary mask for the whole image. See code for details")
    
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")
    parser.add_argument("--total_iters", type=int,  default=60000, help="")
    parser.add_argument("--save_every_iters", type=int,  default=10000, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "true",  default=False, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")

    ######################################
    parser.add_argument("--lambda_eps", type=float, default=1.0, help="weight for noise prediction loss")
    parser.add_argument("--lambda_img", type=float, default=0.1, help="global weight for L_img")
    parser.add_argument("--lambda_pix", type=float, default=1.0, help="weight for full-image L1 loss inside L_img")
    parser.add_argument("--lambda_perc", type=float, default=0.1, help="weight for perceptual loss inside L_img")
    parser.add_argument("--lambda_sem", type=float, default=1.0, help="weight for sem white-region masked L1 loss inside L_img")
    parser.add_argument("--lambda_bg", type=float, default=0.5, help="weight for background suppression loss outside sem white region inside L_img")
    parser.add_argument("--lambda_boundary", type=float, default=1.0, help="weight for boundary-band masked L1 loss inside L_img")
    parser.add_argument("--use_loss_img", default=True, type=lambda x:x.lower() == "true", help="enable or disable the whole L_img branch")
    parser.add_argument("--use_loss_img_pix", default=True, type=lambda x:x.lower() == "true", help="enable full-image L1 term inside L_img")
    parser.add_argument("--use_loss_img_perc", default=True, type=lambda x:x.lower() == "true", help="enable perceptual term inside L_img")
    parser.add_argument("--use_loss_img_sem", default=True, type=lambda x:x.lower() == "true", help="enable sem white-region masked L1 term inside L_img")
    parser.add_argument("--use_loss_img_bg", default=False, type=lambda x:x.lower() == "true", help="enable background suppression loss outside sem white region inside L_img")
    parser.add_argument("--use_loss_img_boundary", default=False, type=lambda x:x.lower() == "true", help="enable boundary-band masked L1 term inside L_img")
    parser.add_argument("--use_loss_img_t_weight", default=True, type=lambda x:x.lower() == "true", help="enable timestep weight w(t) for L_img")
    parser.add_argument("--image_loss_weight_power", type=float, default=1.0, help="power for w(t)")
    parser.add_argument("--boundary_kernel_size", type=int, default=5, help="odd kernel size used to build sem boundary band")
    parser.add_argument("--boundary_dilate_iter", type=int, default=1, help="number of dilation/erosion iterations used to build sem boundary band")

    parser.add_argument("--decoder_ref_image_path", type=str, default=None, help="reference image path used to extract high-frequency condition for decoder mask")
    parser.add_argument("--decoder_ref_image_size", type=int, default=256, help="resize the decoder reference image to this size before training")
    parser.add_argument("--decoder_ref_image_repeat_to_batch", default=True, type=lambda x:x.lower() == "true", help="repeat one reference image to current batch size")
    parser.add_argument("--decoder_train_full", default=True, type=lambda x:x.lower() == "true", help="train decoder and post_quant_conv together with decoder mask modules")
    parser.add_argument("--decoder_debug_every", type=int, default=50, help="print decoder mask grad/debug info every N iters")
    parser.add_argument("--log_loss_every", type=int, default=50, help="print total/eps/img losses every N iters")

    parser.add_argument("--save_attn_heatmap_every", type=int, default=5000, help="save GatedSelfAttentionDense attention heatmap panels every N iters; 0 disables saving")
    parser.add_argument("--save_attn_heatmap_layers", default=False, type=lambda x:x.lower() == "true", help="kept for compatibility; high-res attention visualization now saves panel only")
    parser.add_argument("--save_attn_heatmap_max_samples", type=int, default=1, help="maximum number of samples per save step for attention heatmaps")
    parser.add_argument("--use_loss_attn", default=True, type=lambda x:x.lower() == "true", help="enable or disable gated self-attention alignment loss")
    parser.add_argument("--lambda_attn", type=float, default=1.0, help="weight for gated self-attention alignment loss")
    parser.add_argument("--attn_loss_normalize", default=True, type=lambda x:x.lower() == "true", help="normalize averaged attention heatmap before computing attention loss")
    parser.add_argument("--attn_supervision_size", type=int, default=256, help="attention supervision resolution; set <=0 to use native grounding mask size")
    parser.add_argument("--attn_use_native_grounding_mask", default=True, type=lambda x:x.lower() == "true", help="use original grounding_extra_input resolution as attention supervision target")
    parser.add_argument("--attn_supervision_interp", type=str, default="bicubic", help="upsampling mode for attention heatmaps: nearest|bilinear|bicubic")
    parser.add_argument("--attn_save_panel_only", default=True, type=lambda x:x.lower() == "true", help="save only the attention panel image")

    parser.add_argument("--use_loss_clip_mask", default=True, type=lambda x:x.lower() == "true", help="enable masked CLIP loss on decoded x0 during training")
    parser.add_argument("--lambda_clip_mask", type=float, default=0.015, help="weight for masked CLIP loss")
    parser.add_argument("--clip_mask_model_path", type=str, default="openai/clip-vit-large-patch14", help="local HuggingFace CLIP folder path, e.g. openai/clip-vit-large-patch14")
    parser.add_argument("--clip_mask_aug_num", type=int, default=8, help="number of differentiable CLIP augmentations per sample")
    parser.add_argument("--clip_mask_aug_p", type=float, default=0.7, help="probability for differentiable CLIP augmentations")
    parser.add_argument("--clip_mask_use_colon_suffix", default=True, type=lambda x:x.lower() == "true", help="use only the caption text after the first colon for CLIP text supervision")
    parser.add_argument("--clip_mask_use_mask", default=True, type=lambda x:x.lower() == "true", help="multiply decoded image by grounding_extra_input-derived binary mask before CLIP")
    parser.add_argument("--clip_mask_log_every", type=int, default=50, help="reserved for CLIP mask loss logging cadence")
    parser.add_argument("--save_first_clip_mask_pred_x0", default=True, type=lambda x:x.lower() == "true", help="save the first masked predicted x0 image/panel for inspection")
    parser.add_argument("--first_clip_mask_save_iter", type=int, default=0, help="training iteration index for saving the first masked predicted x0 debug image")
#########################################
   

    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']


    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()



    config = OmegaConf.load(args.yaml_file) 
    config.update( vars(args) )
    config.total_batch_size = config.batch_size * n_gpu
    if args.inpaint_mode:
        config.model.params.inpaint_mode = True


    trainer = Trainer(config)
    synchronize()
    trainer.start_training()










