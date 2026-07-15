
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torchF
import torchvision.transforms as transforms
import torchvision.transforms.functional as TVF
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def batch_to_device(batch, device):
    """
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(batch_to_device(v, device) for v in batch)
    return batch


device = "cuda" if torch.cuda.is_available() else "cpu"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

MVTEC_DEFECTS = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": ["bent_wire", "combined", "cable_swap", "cut_inner_insulation", "cut_outer_insulation",
               "missing_cable", "missing_wire", "poke_insulation"],
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def natural_key(text):
    parts = re.split(r"(\d+)", str(text))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_image_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return []
    files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda x: natural_key(x.name))
    return [str(p) for p in files]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stem(path):
    return Path(path).stem


def infer_defect_type_from_stem(stem):
    m = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    return m.group(1) if m else stem


def parse_prompt_to_obj_and_defect(prompt):
    prompt = str(prompt).strip().lower()
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
    return obj_name, defect_type


def build_type_to_masks(mask_dir, caption_map, fallback_object_type=None):
    type_to_entries = defaultdict(list)
    files = list_image_files(mask_dir)
    if len(files) == 0:
        raise ValueError(f"mask_dir 中没有图像: {mask_dir}")

    for file in files:
        fname = os.path.basename(file)
        stem = get_stem(file)
        if fname not in caption_map:
            raise KeyError(f"caption.json 中缺少 {fname} 的提示词")
        prompt = caption_map[fname]
        obj_name, defect_type = parse_prompt_to_obj_and_defect(prompt)
        if defect_type is None:
            defect_type = infer_defect_type_from_stem(stem)
        if obj_name is None:
            obj_name = fallback_object_type if fallback_object_type not in {None, ""} else "unknown"

        type_to_entries[defect_type].append({
            "sem": file,
            "prompt": prompt,
            "defect_type": defect_type,
            "obj_name": obj_name,
            "stem": stem,
            "filename": fname,
        })
    return type_to_entries


def allocate_counts(entries, total_per_type):
    if len(entries) == 0:
        return []
    base = total_per_type // len(entries)
    rem = total_per_type % len(entries)
    out = []
    for idx, e in enumerate(entries):
        c = base + (1 if idx < rem else 0)
        if c > 0:
            out.append((e, c))
    return out


def resolve_mvtec_paths(args):
    if args.object_type is None:
        missing = []
        for name in ["mask_dir", "caption_json", "normal_image_dir"]:
            if getattr(args, name, None) in {None, ""}:
                missing.append(name)
        if missing:
            raise ValueError("未提供 --object_type 自动组装路径，且以下参数缺失: " + ", ".join(missing))
        return args

    if args.mvtec_root in {None, ""}:
        raise ValueError("使用 --object_type 时，必须同时提供 --mvtec_root")

    root = Path(args.mvtec_root)
    obj = args.object_type
    test_dir = root / "test" / obj
    normal_dir = root / "normal_images" / obj

    if args.mask_dir in {None, ""}:
        args.mask_dir = str(test_dir / "Ground_truth")
    if args.caption_json in {None, ""}:
        args.caption_json = str(test_dir / "caption.json")
    if args.normal_image_dir in {None, ""}:
        args.normal_image_dir = str(normal_dir)

    if not os.path.isdir(args.mask_dir):
        raise FileNotFoundError(f"自动解析到的 mask_dir 不存在: {args.mask_dir}")
    if not os.path.isfile(args.caption_json):
        raise FileNotFoundError(f"自动解析到的 caption_json 不存在: {args.caption_json}")
    if not os.path.isdir(args.normal_image_dir):
        raise FileNotFoundError(f"自动解析到的 normal_image_dir 不存在: {args.normal_image_dir}")

    print("\n[Auto Paths] 已根据 object_type 自动分配路径:")
    print(f"  object_type             = {obj}")
    print(f"  mask_dir                = {args.mask_dir}")
    print(f"  caption_json            = {args.caption_json}")
    print(f"  normal_image_dir        = {args.normal_image_dir}")
    print(f"  decoder_ref_image_path  = {args.decoder_ref_image_path}")
    return args


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
    h, w = mask_tensor.shape[-2:]
    lh, lw = int(latent_hw[0]), int(latent_hw[1])
    if h == lh and w == lw:
        return (mask_tensor > 0.5).float()
    if h % lh == 0 and w % lw == 0:
        kh, kw = h // lh, w // lw
        return torchF.max_pool2d(mask_tensor, kernel_size=(kh, kw), stride=(kh, kw))
    return torchF.adaptive_max_pool2d(mask_tensor, output_size=(lh, lw))


@torch.no_grad()
def dilate_anomaly_mask_latent(anomaly_mask_latent, dilate_pixels=0):
    dilate_pixels = int(dilate_pixels)
    if dilate_pixels <= 0:
        return (anomaly_mask_latent > 0.5).float()
    kernel = 2 * dilate_pixels + 1
    dilated = torchF.max_pool2d(
        (anomaly_mask_latent > 0.5).float(),
        kernel_size=kernel,
        stride=1,
        padding=dilate_pixels,
    )
    return (dilated > 0.0).float()


@torch.no_grad()
def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path, map_location="cpu")
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config["model"]).to(device).eval()
    autoencoder = instantiate_from_config(config["autoencoder"]).to(device).eval()
    text_encoder = instantiate_from_config(config["text_encoder"]).to(device).eval()
    diffusion = instantiate_from_config(config["diffusion"]).to(device)

    model.load_state_dict(saved_ckpt["model"], strict=False)
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])
    return model, autoencoder, text_encoder, diffusion, config


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

    prompt = str(meta.get("prompt", "")).strip().lower()
    obj_name, defect_type = parse_prompt_to_obj_and_defect(prompt)
    if obj_name is None:
        obj_name = meta.get("obj_name")
    if defect_type is None:
        defect_type = meta.get("defect_type")

    class_id = 0
    if obj_name is not None and defect_type is not None and obj_name in MVTEC_DEFECTS:
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
def prepare_normal_latent_and_blend_mask_batch(
    normal_image_paths,
    mask_path,
    autoencoder,
    latent_hw,
    target_image_size,
    blend_mask_dilate=1,
):
    if len(normal_image_paths) == 0:
        raise ValueError("normal_image_paths 不能为空")

    mask_img_raw = Image.open(mask_path).convert("L")
    normal_tensors = []
    preprocessed_normals = []
    preprocessed_masks = []

    for normal_image_path in normal_image_paths:
        normal_img_raw = Image.open(normal_image_path).convert("RGB")
        normal_img, mask_img = crop_resize_pair(normal_img_raw, mask_img_raw.copy(), out_size=target_image_size)
        image_tensor = TVF.pil_to_tensor(normal_img).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        normal_tensors.append(image_tensor)
        preprocessed_normals.append(normal_img)
        preprocessed_masks.append(mask_img)

    image_tensor = torch.stack(normal_tensors, dim=0).to(device)
    z0 = autoencoder.encode(image_tensor)
    if tuple(z0.shape[-2:]) != tuple(latent_hw):
        raise ValueError(
            f"normal latent spatial size {tuple(z0.shape[-2:])} does not match target latent_hw {tuple(latent_hw)}. "
            f"Please ensure normal image is resized to model.image_size*8 before encoding."
        )

    mask_tensor = TVF.pil_to_tensor(preprocessed_masks[0]).float() / 255.0
    mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0).to(device)
    anomaly_mask_latent = downsample_anomaly_mask_to_latent(mask_tensor, latent_hw=latent_hw)
    anomaly_mask_latent = dilate_anomaly_mask_latent(anomaly_mask_latent, dilate_pixels=blend_mask_dilate)
    anomaly_mask_latent = anomaly_mask_latent.repeat(len(normal_image_paths), 1, 1, 1)
    preserve_mask = (1.0 - anomaly_mask_latent).clamp(0.0, 1.0)
    anomaly_latent_sum = float(anomaly_mask_latent[0].sum().item())
    return z0, preserve_mask, preprocessed_normals, preprocessed_masks, anomaly_latent_sum


@torch.no_grad()
def apply_strict_final_background_replace_batch(samples_fake, normal_img_pils, mask_img_pils):
    if normal_img_pils is None or mask_img_pils is None:
        return samples_fake
    if len(normal_img_pils) != samples_fake.shape[0] or len(mask_img_pils) != samples_fake.shape[0]:
        raise ValueError("strict background replacement 的 normal/mask 数量与 batch 大小不匹配")

    normal_tensors = []
    anomaly_masks = []
    for normal_img_pil, mask_img_pil in zip(normal_img_pils, mask_img_pils):
        normal_tensor = TVF.pil_to_tensor(normal_img_pil).float() / 255.0
        normal_tensor = normal_tensor * 2.0 - 1.0
        normal_tensors.append(normal_tensor)

        mask_tensor = TVF.pil_to_tensor(mask_img_pil).float() / 255.0
        anomaly_mask = (mask_tensor > 0.5).float()
        anomaly_masks.append(anomaly_mask)

    normal_tensor = torch.stack(normal_tensors, dim=0).to(samples_fake.device)
    anomaly_mask = torch.stack(anomaly_masks, dim=0).to(samples_fake.device)
    preserve_mask = 1.0 - anomaly_mask
    return samples_fake * anomaly_mask + normal_tensor * preserve_mask


def build_dataset_dirs(dataset_root, object_name, defect_type):
    object_root = Path(dataset_root) / object_name
    image_dir = object_root / defect_type / "image"
    mask_dir = object_root / defect_type / "mask"
    ensure_dir(str(image_dir))
    ensure_dir(str(mask_dir))
    return object_root, image_dir, mask_dir


def get_existing_defect_counter(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return 0
    max_idx = -1
    for p in image_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
            continue
        m = re.match(r"^(\d+)_", p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def count_existing_images(image_dir):
   
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return 0
    return sum(
        1
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def append_or_create_metadata_csv(csv_path, new_rows):
    import pandas as pd

    csv_path = Path(csv_path)
    new_df = pd.DataFrame(new_rows)
    if csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path)
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
            if "generated_path" in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=["generated_path"], keep="last")
        except Exception as e:
            print(f"[Metadata][WARN] 读取已有 metadata 失败，将只保存本次新记录: {csv_path}, error={e}")
            merged_df = new_df
    else:
        merged_df = new_df

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


class InferenceEngine:
    def __init__(self, args):
        self.args = args
        self.model, self.autoencoder, self.text_encoder, self.diffusion, config = load_ckpt(args.ckpt)

        self.grounding_tokenizer_input = instantiate_from_config(config["grounding_tokenizer_input"])
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input

        self.grounding_downsampler_input = None
        if "grounding_downsampler_input" in config:
            self.grounding_downsampler_input = instantiate_from_config(config["grounding_downsampler_input"])

        config.update(vars(args))
        self.config = OmegaConf.create(config)

        if args.no_plms:
            self.sampler = DDIMSampler(self.diffusion, self.model)
            self.steps = int(getattr(self.config, "ddim_steps", 250))
            self.sampler_name = "ddim"
        else:
            self.sampler = PLMSSampler(self.diffusion, self.model)
            self.steps = int(getattr(self.config, "plms_steps", 50))
            self.sampler_name = "plms"

    @torch.no_grad()
    def generate_batch(self, meta, num_images, generated_image_dir, generated_mask_dir, per_defect_start_idx):
        target_image_size = int(self.model.image_size) * 8
        batch = prepare_batch_sem(meta, num_images, target_image_size=target_image_size)

        context = self.text_encoder.encode([meta["prompt"]] * num_images)
        uc = self.text_encoder.encode(num_images * [""])
        if self.args.negative_prompt is not None:
            uc = self.text_encoder.encode(num_images * [self.args.negative_prompt])

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        grounding_extra_input = None
        if self.grounding_downsampler_input is not None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        decoder_ref_path = meta.get("decoder_ref_image_path")
        decoder_ref_images = load_decoder_reference_image(decoder_ref_path, self.args.decoder_ref_image_size)
        decoder_ref_images = get_decoder_reference_batch(
            decoder_ref_images,
            batch_size=num_images,
            repeat_to_batch=self.args.decoder_ref_image_repeat_to_batch,
        )
        if decoder_ref_images is not None and hasattr(self.autoencoder, "set_decoder_reference_images"):
            self.autoencoder.set_decoder_reference_images(decoder_ref_images)
            print(f"[Infer][DecoderRef] path={decoder_ref_path} shape={tuple(decoder_ref_images.shape)}")
        else:
            print("[Infer][DecoderRef] disabled (no decoder reference image provided)")

        input_dict = dict(
            x=None,
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=None,
            grounding_extra_input=grounding_extra_input,
        )

        normal_image_paths = meta["normal_image_paths"]
        z0, blend_mask, preprocessed_normals, preprocessed_masks, anomaly_latent_sum = prepare_normal_latent_and_blend_mask_batch(
            normal_image_paths=normal_image_paths,
            mask_path=meta["sem"],
            autoencoder=self.autoencoder,
            latent_hw=(self.model.image_size, self.model.image_size),
            target_image_size=target_image_size,
            blend_mask_dilate=getattr(self.config, "blend_mask_dilate", 1),
        )
        print(
            f"[BlendDiffusion] sampler={self.sampler_name} mask={meta['sem']} target_image_size={target_image_size} "
            f"latent={tuple(z0.shape)} preserve_mask={tuple(blend_mask.shape)} "
            f"blend_start_step={getattr(self.config, 'blend_start_step', None)} "
            f"blend_mask_dilate={getattr(self.config, 'blend_mask_dilate', 1)} anomaly_latent_sum={anomaly_latent_sum:.1f}"
        )

        blend_start_step = getattr(self.config, "blend_start_step", None)
        if blend_start_step is not None:
            blend_start_step = int(blend_start_step)

        shape = (num_images, self.model.in_channels, self.model.image_size, self.model.image_size)
        samples_fake_latent = self.sampler.sample(
            S=self.steps,
            shape=shape,
            input=input_dict,
            uc=uc,
            guidance_scale=self.config.guidance_scale,
            mask=blend_mask,
            x0=z0,
            blend_start_step=blend_start_step,
        )
        samples_fake = self.autoencoder.decode(
            samples_fake_latent,
            decoder_ref_images=decoder_ref_images,
            grounding_extra_input=grounding_extra_input,
        )

        if bool(getattr(self.config, "strict_final_background_replace", False)):
            samples_fake = apply_strict_final_background_replace_batch(
                samples_fake=samples_fake,
                normal_img_pils=preprocessed_normals,
                mask_img_pils=preprocessed_masks,
            )
            print("[BlendDiffusion] strict final background replacement enabled")

        rows = []
        mask_stem = meta["stem"]
        for local_i, sample in enumerate(samples_fake):
            image_id = per_defect_start_idx + local_i
            normal_path = normal_image_paths[local_i]
            normal_stem = get_stem(normal_path)
            base_name = f"{image_id:04d}_{mask_stem}_{normal_stem}"
            img_name = f"{base_name}.png"
            mask_name = f"{base_name}{Path(meta['sem']).suffix}"

            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.detach().cpu().numpy().transpose(1, 2, 0) * 255
            sample = Image.fromarray(sample.astype(np.uint8))

            out_img_path = os.path.join(generated_image_dir, img_name)
            out_mask_path = os.path.join(generated_mask_dir, mask_name)
            sample.save(out_img_path)
            shutil.copy2(meta["sem"], out_mask_path)

            rows.append({
                "object_name": meta["obj_name"],
                "defect_type": meta["defect_type"],
                "mask_path": meta["sem"],
                "mask_saved_path": out_mask_path,
                "mask_filename": meta["filename"],
                "prompt": meta["prompt"],
                "generated_path": out_img_path,
                "generated_name": img_name,
                "mask_stem": mask_stem,
                "normal_image_path": normal_path,
                "normal_image_stem": normal_stem,
                "per_defect_index": image_id,
            })
        return rows


def sample_normal_paths(normal_image_dir, k):
    normal_files = list_image_files(normal_image_dir)
    if len(normal_files) == 0:
        raise ValueError(f"normal_image_dir 中没有图像: {normal_image_dir}")
    return [random.choice(normal_files) for _ in range(k)]


def generate_all(args):
    caption_map = load_json(args.caption_json)
    type_to_entries = build_type_to_masks(args.mask_dir, caption_map, fallback_object_type=args.object_type)
    print("发现的缺陷类型与 mask 数量：")
    for defect_type, entries in type_to_entries.items():
        print(f"  - {defect_type}: {len(entries)}")

    dataset_root = ensure_dir(os.path.join(args.folder, args.save_folder_name))

    # 先检查每类缺陷已经生成了多少张，再决定是否需要继续生成。
    # 这样可以用于断点续生成：已有数量 >= --total_per_type 的类别会直接跳过。
    generation_plan = []
    total_target_images = 0
    print("\n[Resume Check] 检查每类缺陷当前已有图片数量:")
    for defect_type in sorted(type_to_entries.keys(), key=natural_key):
        entries = sorted(type_to_entries[defect_type], key=lambda x: natural_key(x["filename"]))
        if not entries:
            continue

        object_name = entries[0].get("obj_name") or args.object_type or "unknown"
        object_root, image_dir, mask_dir = build_dataset_dirs(dataset_root, object_name, defect_type)

        existing_count = count_existing_images(image_dir)
        need_count = max(0, int(args.total_per_type) - int(existing_count))
        defect_counter = get_existing_defect_counter(image_dir)

        if existing_count >= int(args.total_per_type):
            print(
                f"  - object={object_name}, defect_type={defect_type}: "
                f"已有 {existing_count} 张，目标 {args.total_per_type} 张，已满足，跳过生成。"
            )
            continue

        allocations = allocate_counts(entries, need_count)
        real_need_count = sum(target_count for _, target_count in allocations)
        total_target_images += real_need_count
        generation_plan.append({
            "defect_type": defect_type,
            "entries": entries,
            "allocations": allocations,
            "object_name": object_name,
            "object_root": object_root,
            "image_dir": image_dir,
            "mask_dir": mask_dir,
            "existing_count": existing_count,
            "need_count": real_need_count,
            "defect_counter": defect_counter,
        })
        print(
            f"  - object={object_name}, defect_type={defect_type}: "
            f"已有 {existing_count} 张，目标 {args.total_per_type} 张，"
            f"还需生成 {real_need_count} 张，起始编号 {defect_counter}。"
        )

    if total_target_images <= 0:
        print("\n[Generate] 所有缺陷类别的已有图片数量均已达到或超过 --total_per_type，本次不再生成。")
        print(f"[Generate] 数据集根目录: {dataset_root}")
        return None

    # 只有确实需要生成时才加载模型，避免全部跳过时仍占用显存。
    engine = InferenceEngine(args)

    metadata_rows = []
    overall_pbar = tqdm(total=total_target_images, desc="总推理进度", dynamic_ncols=True)
    try:
        for plan in generation_plan:
            defect_type = plan["defect_type"]
            entries = plan["entries"]
            allocations = plan["allocations"]
            object_name = plan["object_name"]
            image_dir = plan["image_dir"]
            mask_dir = plan["mask_dir"]
            defect_counter = plan["defect_counter"]
            type_target_total = plan["need_count"]

            if type_target_total <= 0:
                continue

            print(
                f"\n[Generate] object={object_name}, defect_type={defect_type}, "
                f"masks={len(entries)}, existing={plan['existing_count']}, "
                f"target_total={args.total_per_type}, need_generate={type_target_total}, "
                f"start_idx={defect_counter}"
            )

            type_pbar = tqdm(total=type_target_total, desc=f"{defect_type} 推理进度", dynamic_ncols=True, leave=False)
            try:
                for mask_entry, target_count in allocations:
                    if target_count <= 0:
                        continue
                    print(f"[Generate] {mask_entry['filename']} -> 补生成 {target_count} 张")
                    generated = 0
                    mask_pbar = tqdm(total=target_count, desc=f"mask {mask_entry['filename']}", dynamic_ncols=True, leave=False)
                    try:
                        while generated < target_count:
                            cur_bs = min(args.batch_size, target_count - generated)
                            normal_image_paths = sample_normal_paths(args.normal_image_dir, cur_bs)
                            meta = dict(mask_entry)
                            meta["decoder_ref_image_path"] = args.decoder_ref_image_path
                            meta["normal_image_paths"] = normal_image_paths
                            if meta.get("obj_name") in {None, "unknown"} and args.object_type not in {None, ""}:
                                meta["obj_name"] = args.object_type

                            rows = engine.generate_batch(
                                meta=meta,
                                num_images=cur_bs,
                                generated_image_dir=str(image_dir),
                                generated_mask_dir=str(mask_dir),
                                per_defect_start_idx=defect_counter,
                            )
                            defect_counter += cur_bs
                            generated += cur_bs
                            metadata_rows.extend(rows)

                            mask_pbar.update(cur_bs)
                            type_pbar.update(cur_bs)
                            overall_pbar.update(cur_bs)
                    finally:
                        mask_pbar.close()
            finally:
                type_pbar.close()
    finally:
        overall_pbar.close()

    if not metadata_rows:
        raise RuntimeError("本次计划需要生成图像，但 metadata_rows 为空，请检查推理过程是否正常。")

    object_names = sorted({row["object_name"] for row in metadata_rows}, key=natural_key)
    for object_name in object_names:
        object_root = Path(dataset_root) / object_name
        object_rows = [row for row in metadata_rows if row["object_name"] == object_name]
        meta_csv = object_root / "generated_metadata.csv"
        append_or_create_metadata_csv(meta_csv, object_rows)
        print(f"[Generate] {object_name} metadata 已追加/保存到: {meta_csv}")

    all_meta_csv = os.path.join(dataset_root, "generated_metadata_all_objects.csv")
    append_or_create_metadata_csv(all_meta_csv, metadata_rows)
    print(f"\n[Generate] 本次新增生成 {len(metadata_rows)} 张，总 metadata 已追加/保存到: {all_meta_csv}")
    print(f"[Generate] 数据集根目录: {dataset_root}")
    return all_meta_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./OUTPUT/OUTPUT_Tile_第四阶段/checkpoint_latest.pth")
    parser.add_argument("--folder", type=str, default="../MVTec指标测量/mvtec_batch_generation_samples")
    parser.add_argument("--save_folder_name", type=str, default="generated_samples")

    parser.add_argument("--mvtec_root", type=str, default="DATA/Mvtec_per-object", help="Mvtec_per-object 根目录")
    parser.add_argument("--object_type", type=str, default="tile", choices=sorted(MVTEC_DEFECTS.keys()), help="物品类型，提供后自动解析路径")

    parser.add_argument("--mask_dir", type=str, default=None, help="test/<object>/Ground_truth；若提供 --object_type 可省略")
    parser.add_argument("--caption_json", type=str, default=None, help="test/<object>/caption.json；若提供 --object_type 可省略")
    parser.add_argument("--normal_image_dir", type=str, default=None, help="normal_images/<object>；若提供 --object_type 可省略")

    parser.add_argument("--decoder_ref_image_path", type=str, default=None, help="可选：decoder reference 图像路径，不做自动解析")
    parser.add_argument("--decoder_ref_image_size", type=int, default=256)
    parser.add_argument("--decoder_ref_image_repeat_to_batch", type=str2bool, default=True)

    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_plms", action="store_true", help="使用 DDIM；默认使用 PLMS")
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--strict_final_background_replace", type=str2bool, default=False)
    parser.add_argument("--blend_start_step", type=int, default=50)
    parser.add_argument("--blend_mask_dilate", type=int, default=0)

    parser.add_argument("--total_per_type", type=int, default=2000, help="每种缺陷类型总生成数量")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args = resolve_mvtec_paths(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generate_all(args)


if __name__ == "__main__":
    main()
