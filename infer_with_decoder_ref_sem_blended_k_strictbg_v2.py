import argparse
import os
import re
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms.functional as TVF
import torchvision.transforms as transforms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from trainer import batch_to_device


device = "cuda"


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def list_files(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    file_list.sort()
    return file_list


def alpha_generator(length, type=None):
    if type is None:
        type = [1, 0, 0]
    assert len(type) == 3
    assert abs(type[0] + type[1] + type[2] - 1) < 1e-8

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length
    assert len(alphas) == length
    return alphas


@torch.no_grad()
def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path, map_location="cpu")
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    model.load_state_dict(saved_ckpt['model'], strict=False)
    autoencoder.load_state_dict(saved_ckpt['autoencoder'])
    text_encoder.load_state_dict(saved_ckpt['text_encoder'])
    diffusion.load_state_dict(saved_ckpt['diffusion'])
    return model, autoencoder, text_encoder, diffusion, config


def crop_resize_pair(image_pil, mask_pil, out_size):
    if image_pil.size != mask_pil.size:
        raise ValueError(f"normal image size {image_pil.size} != mask size {mask_pil.size}")
    crop_size = min(image_pil.size)
    image_pil = TVF.center_crop(image_pil, crop_size)
    mask_pil = TVF.center_crop(mask_pil, crop_size)
    image_pil = image_pil.resize((out_size, out_size), Image.BICUBIC)
    mask_pil = mask_pil.resize((out_size, out_size), Image.NEAREST)
    return image_pil, mask_pil


def downsample_anomaly_mask_to_latent(mask_tensor, latent_hw):
    """
    Downsample anomaly mask to latent resolution using max-pooling style reduction.
    This is more defect-friendly than nearest interpolation for tiny anomalies.
    """
    h, w = mask_tensor.shape[-2:]
    lh, lw = int(latent_hw[0]), int(latent_hw[1])

    if h == lh and w == lw:
        return (mask_tensor > 0.5).float()

    if h % lh == 0 and w % lw == 0:
        kh, kw = h // lh, w // lw
        return F.max_pool2d(mask_tensor, kernel_size=(kh, kw), stride=(kh, kw))

    # fallback for non-integer ratios: still use max-pooling style aggregation
    return F.adaptive_max_pool2d(mask_tensor, output_size=(lh, lw))


@torch.no_grad()
def dilate_anomaly_mask_latent(anomaly_mask_latent, dilate_pixels=0):
    dilate_pixels = int(dilate_pixels)
    if dilate_pixels <= 0:
        return (anomaly_mask_latent > 0.5).float()

    kernel = 2 * dilate_pixels + 1
    dilated = F.max_pool2d(
        (anomaly_mask_latent > 0.5).float(),
        kernel_size=kernel,
        stride=1,
        padding=dilate_pixels,
    )
    return (dilated > 0.0).float()


@torch.no_grad()
def prepare_normal_latent_and_blend_mask(
    normal_image_path,
    mask_path,
    autoencoder,
    batch_size,
    latent_hw,
    target_image_size,
    blend_mask_dilate=1,
):
    normal_img = Image.open(normal_image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")
    normal_img, mask_img = crop_resize_pair(normal_img, mask_img, out_size=target_image_size)

    image_tensor = TVF.pil_to_tensor(normal_img).float() / 255.0
    image_tensor = (image_tensor - 0.5) / 0.5
    image_tensor = image_tensor.unsqueeze(0).to(device)

    mask_tensor = TVF.pil_to_tensor(mask_img).float() / 255.0
    mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0).to(device)  # [1,1,H,W]

    z0 = autoencoder.encode(image_tensor)
    if z0.shape[0] != batch_size:
        z0 = z0.repeat(batch_size, 1, 1, 1)

    if tuple(z0.shape[-2:]) != tuple(latent_hw):
        raise ValueError(
            f"normal latent spatial size {tuple(z0.shape[-2:])} does not match target latent_hw {tuple(latent_hw)}. "
            f"Please ensure normal image is resized to model.image_size*8 before encoding."
        )

    anomaly_mask_latent = downsample_anomaly_mask_to_latent(mask_tensor, latent_hw=latent_hw)
    anomaly_mask_latent = dilate_anomaly_mask_latent(
        anomaly_mask_latent,
        dilate_pixels=blend_mask_dilate,
    )

    if anomaly_mask_latent.shape[0] != batch_size:
        anomaly_mask_latent = anomaly_mask_latent.repeat(batch_size, 1, 1, 1)

    preserve_mask = (1.0 - anomaly_mask_latent).clamp(0.0, 1.0)
    anomaly_latent_sum = float(anomaly_mask_latent[0].sum().item())
    return z0, preserve_mask, normal_img, mask_img, anomaly_latent_sum


@torch.no_grad()
def load_decoder_reference_image(path, image_size=256):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"decoder_ref_image_path not found: {path}")
    image = Image.open(path).convert("RGB")
    if image_size is not None and int(image_size) > 0:
        image = image.resize((int(image_size), int(image_size)), resample=Image.BICUBIC)
    tensor = TVF.pil_to_tensor(image).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def get_decoder_reference_batch(ref_tensor, batch_size=1, repeat_to_batch=True):
    if ref_tensor is None:
        return None
    if repeat_to_batch and ref_tensor.size(0) == 1 and batch_size > 1:
        return ref_tensor.repeat(batch_size, 1, 1, 1)
    if ref_tensor.size(0) != batch_size:
        raise ValueError(
            f"decoder reference batch mismatch: ref_batch={ref_tensor.size(0)}, batch_size={batch_size}. "
            f"Set --decoder_ref_image_repeat_to_batch or use batch_size=1."
        )
    return ref_tensor


def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * np.tile(colors[label], (labelmap.shape[0], labelmap.shape[1], 1))
    return labelmap_rgb


@torch.no_grad()
def prepare_batch_sem(meta, batch=1, target_image_size=512):
    pil_to_tensor = transforms.PILToTensor()

    sem_src = meta["sem"]
    if isinstance(sem_src, str):
        sem_img = Image.open(sem_src).convert("L")
    elif isinstance(sem_src, Image.Image):
        sem_img = sem_src.convert("L")
    else:
        raise TypeError(f"meta['sem'] 类型不支持: {type(sem_src)}，应为路径(str)或 PIL.Image")

    crop_size = min(sem_img.size)
    sem_img = TVF.center_crop(sem_img, crop_size)
    sem_img = sem_img.resize((target_image_size, target_image_size), Image.NEAREST)
    sem = pil_to_tensor(sem_img)[0, :, :].float()
    sem = (sem > 127).float()

    MVTEC_DEFECTS = {
        "bottle": ["broken_large", "broken_small", "contamination"],
        "cable": ["bent_wire", "combined", "cable_swap", "cut_inner_insulation", "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"],
        "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
        "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
        "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
        "hazelnut": ["crack", "cut", "hole", "print"],
        "leather": ["color", "cut", "fold", "glue", "poke"],
        "metal_nut": ["bent", "color", "flip", "scratch"],
        "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
        "screw": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
        "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
        "toothbrush": ["defective"],
        "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
        "wood": ["color", "combined", "hole", "liquid", "scratch"],
        "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "squeezed_teeth", "split_teeth"],
    }

    prompt = str(meta.get("prompt", "")).strip().lower()
    m = re.search(r"a photo of\s+(?:a|an|the\s+)?(.*?)\s+with\s+(.*?)\s+defect", prompt)
    obj_phrase = m.group(1).strip() if m else ""
    defect_phrase = m.group(2).strip() if m else ""

    obj_name = None
    if obj_phrase:
        for name in MVTEC_DEFECTS.keys():
            if obj_phrase == name.replace("_", " ") or obj_phrase == name:
                obj_name = name
                break
    if obj_name is None:
        for name in MVTEC_DEFECTS.keys():
            if name in prompt or name.replace("_", " ") in prompt:
                obj_name = name
                break

    defect_type = None
    if obj_name is not None:
        defects = MVTEC_DEFECTS[obj_name]
        if defect_phrase:
            cand = defect_phrase.replace(" ", "_")
            if cand in defects:
                defect_type = cand
            else:
                for d in defects:
                    if d.replace("_", " ") in defect_phrase:
                        defect_type = d
                        break
        else:
            for d in defects:
                if d in prompt or d.replace("_", " ") in prompt:
                    defect_type = d
                    break

    class_id = 0
    if obj_name is not None and defect_type is not None:
        sorted_defects = sorted(MVTEC_DEFECTS[obj_name])
        defect_to_idx = {"background": 0}
        for i, d in enumerate(sorted_defects, start=1):
            defect_to_idx[d] = i
        class_id = defect_to_idx.get(defect_type, 0)

    sem_idx = (sem * float(class_id)).long()
    num_channels = 152
    if class_id >= num_channels:
        sem_idx = sem_idx * 0

    input_label = torch.zeros(num_channels, target_image_size, target_image_size)
    sem_oh = input_label.scatter_(0, sem_idx.unsqueeze(0), 1.0)
    out = {
        "sem": sem_oh.unsqueeze(0).repeat(batch, 1, 1, 1),
        "mask": torch.ones(batch, 1),
    }
    return batch_to_device(out, device)


@torch.no_grad()
def save_blend_debug(normal_img, mask_img, output_folder, base_name):
    if normal_img is None or mask_img is None:
        return
    normal_img.save(os.path.join(output_folder, f"{base_name}_normal_preprocessed.png"))
    mask_img.save(os.path.join(output_folder, f"{base_name}_mask_preprocessed.png"))


@torch.no_grad()
def apply_strict_final_background_replace(samples_fake, normal_img_pil, mask_img_pil):
    """
    samples_fake: [B, 3, H, W], range [-1, 1]
    normal_img_pil: preprocessed normal image, same spatial size as output image
    mask_img_pil: preprocessed anomaly mask, white=anomaly/edit region, black=background
    return: blended output in [-1, 1]
    """
    if normal_img_pil is None or mask_img_pil is None:
        return samples_fake

    normal_tensor = TVF.pil_to_tensor(normal_img_pil).float().to(samples_fake.device) / 255.0
    normal_tensor = normal_tensor * 2.0 - 1.0
    normal_tensor = normal_tensor.unsqueeze(0)

    mask_tensor = TVF.pil_to_tensor(mask_img_pil).float().to(samples_fake.device) / 255.0
    anomaly_mask = (mask_tensor > 0.5).float().unsqueeze(0)
    preserve_mask = 1.0 - anomaly_mask

    if normal_tensor.shape[0] != samples_fake.shape[0]:
        normal_tensor = normal_tensor.repeat(samples_fake.shape[0], 1, 1, 1)
    if anomaly_mask.shape[0] != samples_fake.shape[0]:
        anomaly_mask = anomaly_mask.repeat(samples_fake.shape[0], 1, 1, 1)
        preserve_mask = preserve_mask.repeat(samples_fake.shape[0], 1, 1, 1)

    return samples_fake * anomaly_mask + normal_tensor * preserve_mask


@torch.no_grad()
def run(meta, config, starting_noise=None):
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

    config.update(vars(args))
    config = OmegaConf.create(config)

    decoder_ref_path = meta.get("decoder_ref_image_path", getattr(config, "decoder_ref_image_path", None))
    decoder_ref_size = int(meta.get("decoder_ref_image_size", getattr(config, "decoder_ref_image_size", 256)))
    decoder_ref_repeat_to_batch = bool(meta.get("decoder_ref_image_repeat_to_batch", getattr(config, "decoder_ref_image_repeat_to_batch", True)))
    decoder_ref_images = load_decoder_reference_image(decoder_ref_path, decoder_ref_size)
    decoder_ref_images = get_decoder_reference_batch(decoder_ref_images, batch_size=config.batch_size, repeat_to_batch=decoder_ref_repeat_to_batch)
    if decoder_ref_images is not None and hasattr(autoencoder, "set_decoder_reference_images"):
        autoencoder.set_decoder_reference_images(decoder_ref_images)
        print(f"[Infer][DecoderRef] path={decoder_ref_path} shape={tuple(decoder_ref_images.shape)}")
    else:
        print("[Infer][DecoderRef] disabled (no decoder reference image provided)")

    target_image_size = int(model.image_size) * 8
    batch = prepare_batch_sem(meta, config.batch_size, target_image_size=target_image_size)
    context = text_encoder.encode([meta["prompt"]] * config.batch_size)
    uc = text_encoder.encode(config.batch_size * [""])
    if args.negative_prompt is not None:
        uc = text_encoder.encode(config.batch_size * [args.negative_prompt])

    if config.no_plms:
        sampler = DDIMSampler(diffusion, model)
        steps = int(getattr(config, "ddim_steps", 250))
        sampler_name = "ddim"
    else:
        sampler = PLMSSampler(diffusion, model)
        steps = int(getattr(config, "plms_steps", 50))
        sampler_name = "plms"

    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input is not None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input_dict = dict(
        x=starting_noise,
        timesteps=None,
        context=context,
        grounding_input=grounding_input,
        inpainting_extra_input=None,
        grounding_extra_input=grounding_extra_input,
    )

    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

    blend_mask = None
    z0 = None
    preprocessed_normal = None
    preprocessed_mask = None
    normal_image_path = meta.get("normal_image_path", None)
    if normal_image_path is not None:
        z0, blend_mask, preprocessed_normal, preprocessed_mask, anomaly_latent_sum = prepare_normal_latent_and_blend_mask(
            normal_image_path=normal_image_path,
            mask_path=meta["sem"],
            autoencoder=autoencoder,
            batch_size=config.batch_size,
            latent_hw=(model.image_size, model.image_size),
            target_image_size=target_image_size,
            blend_mask_dilate=getattr(config, "blend_mask_dilate", 1),
        )
        print(
            f"[BlendDiffusion] sampler={sampler_name} normal_image={normal_image_path} mask={meta['sem']} "
            f"target_image_size={target_image_size} latent={tuple(z0.shape)} preserve_mask={tuple(blend_mask.shape)} "
            f"blend_start_step={getattr(config, 'blend_start_step', None)} blend_mask_dilate={getattr(config, 'blend_mask_dilate', 1)} "
            f"anomaly_latent_sum={anomaly_latent_sum:.1f}"
        )

    blend_start_step = getattr(config, "blend_start_step", None)
    if blend_start_step is not None:
        blend_start_step = int(blend_start_step)

    if blend_start_step is not None and z0 is None:
        raise ValueError("blend_start_step is set but normal_image_path/x0 is missing. A normal reference image is required to initialize x_k.")

    samples_fake_latent = sampler.sample(
        S=steps,
        shape=shape,
        input=input_dict,
        uc=uc,
        guidance_scale=config.guidance_scale,
        mask=blend_mask,
        x0=z0,
        blend_start_step=blend_start_step,
    )
    samples_fake = autoencoder.decode(
        samples_fake_latent,
        decoder_ref_images=decoder_ref_images,
        grounding_extra_input=grounding_extra_input,
    )

    strict_bg_enabled = bool(getattr(config, "strict_final_background_replace", False))
    if strict_bg_enabled:
        if preprocessed_normal is not None and preprocessed_mask is not None:
            samples_fake = apply_strict_final_background_replace(
                samples_fake=samples_fake,
                normal_img_pil=preprocessed_normal,
                mask_img_pil=preprocessed_mask,
            )
            print("[BlendDiffusion] strict final background replacement enabled")
        else:
            print("[BlendDiffusion] strict final background replacement requested but skipped (missing normal image or mask)")

    output_folder = os.path.join(args.folder, meta["save_folder_name"])
    os.makedirs(output_folder, exist_ok=True)

    start = len([f for f in os.listdir(output_folder) if f.endswith('.png')])
    image_ids = list(range(start, start + config.batch_size))
    base_name = os.path.splitext(os.path.basename(meta["sem"]))[0]

    if args.save_blend_debug and preprocessed_normal is not None:
        save_blend_debug(preprocessed_normal, preprocessed_mask, output_folder, base_name)

    for image_id, sample in zip(image_ids, samples_fake):
        strict_tag = "strictbg" if bool(getattr(config, "strict_final_background_replace", False)) else "nostrictbg"
        blend_k = getattr(config, "blend_start_step", None)
        blend_tag = f"k{int(blend_k):03d}" if blend_k is not None else "kfull"
        dilate_tag = f"d{int(getattr(config, 'blend_mask_dilate', 1))}"
        img_name = f"{base_name}_{sampler_name}_{blend_tag}_{dilate_tag}_{strict_tag}_fake_{image_id:03d}.png"
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        sample.save(os.path.join(output_folder, img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="generation_samples", help="root folder for output")
    parser.add_argument("--mask_dir", type=str, default="./Infer_DATA/mask")
    parser.add_argument("--prompt", type=str, default="A photo of hazelnut with hole defect: a punctured hole is visible, indicating missing material.")
    parser.add_argument("--save_folder_name", type=str, default="mvtec_im2img_blend-diffusion")
    parser.add_argument("--guidance_scale", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_plms", action="store_true", help="use DDIM; default is PLMS when flag not set")
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--decoder_ref_image_path", type=str, default=None, help="reference image path for decoder weight mask high-frequency condition")
    parser.add_argument("--decoder_ref_image_size", type=int, default=256, help="resize size for decoder reference image")
    parser.add_argument("--decoder_ref_image_repeat_to_batch", type=str2bool, default=True, help="repeat one decoder reference image to full batch (true/false)")

    parser.add_argument("--normal_image_path", type=str, default="./DATA/Mvtec_per-object/normal_images/hazelnut/normal_hazelnut_046.png", help="path to the normal image y0 used in blended diffusion")
    parser.add_argument("--save_blend_debug", type=str2bool, default=True, help="save preprocessed normal image and mask for debugging")
    parser.add_argument("--strict_final_background_replace", type=str2bool, default=False, help="replace background outside anomaly mask with the preprocessed normal image after decoding")
    parser.add_argument("--blend_start_step", type=int, default=125, help="optional sampling start index k in the sampler schedule; when set, initialize x_k=q(x0,k) and only sample from k to 0")
    parser.add_argument("--blend_mask_dilate", type=int, default=0, help="dilate anomaly mask by this many latent pixels before blended diffusion")
    parser.add_argument("--ckpt", type=str, default="./OUTPUT/OUTPUT_Hazelnut_第四阶段/checkpoint_latest.pth")
    args = parser.parse_args()


    files = list_files(args.mask_dir)
    meta_list = []
    for file in files:
        meta_list.append(dict(
            ckpt=args.ckpt,
            prompt=args.prompt,
            sem=file,
            alpha_type=[0.7, 0, 0.3],
            save_folder_name=args.save_folder_name,
            decoder_ref_image_path=args.decoder_ref_image_path,
            decoder_ref_image_size=args.decoder_ref_image_size,
            decoder_ref_image_repeat_to_batch=args.decoder_ref_image_repeat_to_batch,
            normal_image_path=args.normal_image_path,
        ))

    starting_noise = None
    for meta in meta_list:
        run(meta, args, starting_noise)
