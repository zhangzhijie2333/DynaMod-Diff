import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.utils.data
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[-1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def exist_in(short_str, list_of_string):
    for string in list_of_string:
        if short_str in string:
            return True
    return False


def clean_files(image_files, normal_files):
    """
    Not sure why some images do not have normal map annotations, thus delete these images from list.

    The implementation here is inefficient .....
    """
    new_image_files = []

    for image_file in image_files:
        image_file_basename = os.path.basename(image_file).split('.')[0]
        if exist_in(image_file_basename, normal_files):
            new_image_files.append(image_file)
    image_files = new_image_files

    # a sanity check
    for image_file, normal_file in zip(image_files, normal_files):
        image_file_basename = os.path.basename(image_file).split('.')[0]
        normal_file_basename = os.path.basename(normal_file).split('.')[0]
        assert image_file_basename == normal_file_basename[:-7]

    return image_files, normal_files


class SemanticDataset():
    def __init__(self, image_rootdir, sem_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
        self.image_rootdir = image_rootdir
        self.sem_rootdir = sem_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_flip = random_flip

        # Image and normal files
        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['bmp'])
        image_files.sort()
        sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=['png'])
        sem_files.sort()

        self.image_files = image_files
        self.sem_files = sem_files

        # Open caption json
        with open(caption_path, 'r') as f:
            self.image_filename_to_caption_mapping = json.load(f)

        assert len(self.image_files) == len(self.sem_files) == len(self.image_filename_to_caption_mapping)
        self.pil_to_tensor = transforms.PILToTensor()

    def total_images(self):
        return len(self)

    def __getitem__(self, index):
        image_path = self.image_files[index]

        out = {}

        out['id'] = index
        image = Image.open(image_path).convert("RGB")
        sem = Image.open(self.sem_files[index]).convert("L")  # semantic class index 0,1,2,3,4 in uint8 representation

        assert image.size == sem.size

        # - - - - - center_crop, resize and random_flip - - - - - - #
        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize((self.image_size, self.image_size))

        sem = TF.center_crop(sem, crop_size)
        sem = sem.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            sem = ImageOps.mirror(sem)

        sem = self.pil_to_tensor(sem)[0, :, :]

        if 'In' in os.path.basename(image_path):
            sem = sem / 255
        elif 'Pa' in os.path.basename(image_path):
            sem = sem / 255 * 2
        elif 'Sc' in os.path.basename(image_path):
            sem = sem / 255 * 3

        input_label = torch.zeros(152, self.image_size, self.image_size)
        sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

        out['image'] = (self.pil_to_tensor(image).float() / 255 - 0.5) / 0.5
        out['sem'] = sem
        out['mask'] = torch.tensor(1.0)

        # -------------------- caption ------------------- #
        if random.uniform(0, 1) < self.prob_use_caption:
            if 'In' in os.path.basename(image_path):
                out["caption"] = 'inclusion'
            elif 'Pa' in os.path.basename(image_path):
                out["caption"] = 'patches'
            elif 'Sc' in os.path.basename(image_path):
                out["caption"] = 'scratches'
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.image_files)


class MvtecSemanticDataset(torch.utils.data.Dataset):

    MVTEC_DEFECTS = {
        "bottle":      ["broken_large", "broken_small", "contamination"],
        "cable":       ["bent_wire", "combined", "cable_swap", "cut_inner_insulation", "cut_outer_insulation",
                        "missing_cable", "missing_wire", "poke_insulation"],
        "capsule":     ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
        "carpet":      ["color", "cut", "hole", "metal_contamination", "thread"],
        "grid":        ["bent", "broken", "glue", "metal_contamination", "thread"],
        "hazelnut":    ["crack", "cut", "hole", "print"],
        "leather":     ["color", "cut", "fold", "glue", "poke"],
        "metal_nut":   ["bent", "color", "flip", "scratch"],
        "pill":        ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
        "screw":       ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
        "tile":        ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
        "toothbrush":  ["defective"],
        "transistor":  ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
        "wood":        ["color", "combined", "hole", "liquid", "scratch"],
        "zipper":      ["broken_teeth", "combined", "fabric_border", "fabric_interior",
                        "rough", "squeezed_teeth", "split_teeth"],
    }

    def __init__(
        self,
        image_rootdir: str,
        sem_rootdir: str,
        caption_path: str,
        prob_use_caption: float = 1.0,
        image_size: int = 512,
        random_flip: bool = False,
        num_channels: int = 152,
        obj_name: Optional[str] = None,
        # ===== paired augmentation =====
        paired_aug_enable: bool = False,
        paired_aug_prob: float = 0.8,
        paired_aug_crop_prob: float = 0.8,
        paired_aug_crop_scale_min: float = 0.75,
        paired_aug_crop_scale_max: float = 1.0,
        paired_aug_crop_ratio_min: float = 0.9,
        paired_aug_crop_ratio_max: float = 1.1,
        paired_aug_translate_prob: float = 0.8,
        paired_aug_translate_frac: float = 0.08,
        paired_aug_rotate_prob: float = 0.8,
        paired_aug_rotate_deg: float = 12.0,
        paired_aug_min_fg_keep_ratio: float = 0.97,
        paired_aug_max_tries: int = 20,
        paired_aug_apply_to_normal: bool = True,
        debug_aug: bool = False,
    ):
        super().__init__()
        self.image_rootdir = image_rootdir
        self.sem_rootdir = sem_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.num_channels = int(num_channels)

        self.paired_aug_enable = bool(paired_aug_enable)
        self.paired_aug_prob = float(paired_aug_prob)
        self.paired_aug_crop_prob = float(paired_aug_crop_prob)
        self.paired_aug_crop_scale_min = float(paired_aug_crop_scale_min)
        self.paired_aug_crop_scale_max = float(paired_aug_crop_scale_max)
        self.paired_aug_crop_ratio_min = float(paired_aug_crop_ratio_min)
        self.paired_aug_crop_ratio_max = float(paired_aug_crop_ratio_max)
        self.paired_aug_translate_prob = float(paired_aug_translate_prob)
        self.paired_aug_translate_frac = float(paired_aug_translate_frac)
        self.paired_aug_rotate_prob = float(paired_aug_rotate_prob)
        self.paired_aug_rotate_deg = float(paired_aug_rotate_deg)
        self.paired_aug_min_fg_keep_ratio = float(paired_aug_min_fg_keep_ratio)
        self.paired_aug_max_tries = int(paired_aug_max_tries)
        self.paired_aug_apply_to_normal = bool(paired_aug_apply_to_normal)
        self.debug_aug = bool(debug_aug)

        if obj_name is None:
            obj_name = Path(image_rootdir).resolve().parent.name
        self.obj_name = obj_name

        if self.obj_name not in self.MVTEC_DEFECTS:
            raise ValueError(
                f"[MvtecSemanticDataset] 无法识别 obj_name={self.obj_name}。\n"
                f"请确认 image_rootdir={image_rootdir} 是否形如 .../<类别>/Source_Images，\n"
                f"或在初始化时显式传入 obj_name。"
            )

        self.DEFECT_TO_IDX = {"background": 0}
        for i, defect in enumerate(sorted(self.MVTEC_DEFECTS[self.obj_name]), start=1):
            self.DEFECT_TO_IDX[defect] = i
        self.NUM_DEFECT_CLASSES = 1 + len(self.MVTEC_DEFECTS[self.obj_name])

        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=["png"])
        sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=["png"])
        image_files.sort()
        sem_files.sort()

        sem_map = {os.path.basename(p): p for p in sem_files}

        paired_images, paired_sems = [], []
        miss = 0
        for img_p in image_files:
            bn = os.path.basename(img_p)
            paired_images.append(img_p)
            if bn in sem_map:
                paired_sems.append(sem_map[bn])
            else:
                paired_sems.append(None)
                miss += 1

        if miss > 0:
            print(f"[WARNING] {self.obj_name}: 有 {miss} 张图找不到对应 mask，将使用全0 mask（background）。")

        self.image_files = paired_images
        self.sem_files = paired_sems

        with open(caption_path, "r", encoding="utf-8") as f:
            self.image_filename_to_caption_mapping = json.load(f)

        self.pil_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.image_files)

    def total_images(self):
        return len(self)

    def _parse_defect_from_filename(self, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0].lower()

        if base.startswith("normal_"):
            return "background"

        parts = base.split("_")
        if len(parts) < 2:
            return "background"

        if parts[-1].isdigit():
            parts = parts[:-1]
        if len(parts) == 0:
            return "background"

        if parts[0] == self.obj_name:
            parts = parts[1:]
        if len(parts) == 0:
            return "background"

        return "_".join(parts)

    def _center_crop_resize(self, img: Image.Image, is_mask: bool) -> Image.Image:
        crop_size = min(img.size)
        img = TF.center_crop(img, crop_size)
        if is_mask:
            img = img.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        else:
            img = img.resize((self.image_size, self.image_size), resample=Image.LANCZOS)
        return img

    def _estimate_fill_rgb(self, image: Image.Image) -> Tuple[int, int, int]:
        t = TF.to_tensor(image)
        mean_rgb = (t.mean(dim=(1, 2)) * 255.0).round().clamp(0, 255).to(torch.int64)
        return int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])

    def _mask_bbox(self, mask: Image.Image):
        mask_t = self.pil_to_tensor(mask)[0]
        ys, xs = torch.where(mask_t > 127)
        if ys.numel() == 0:
            return None
        x_min, x_max = int(xs.min().item()), int(xs.max().item())
        y_min, y_max = int(ys.min().item()), int(ys.max().item())
        return x_min, y_min, x_max, y_max

    def _foreground_area(self, mask: Image.Image) -> int:
        mask_t = self.pil_to_tensor(mask)[0]
        return int((mask_t > 127).sum().item())

    def _sample_safe_crop_box(self, width: int, height: int, bbox):
        if bbox is None:
            return None

        x_min, y_min, x_max, y_max = bbox
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1
        area = width * height

        for _ in range(self.paired_aug_max_tries):
            scale = random.uniform(self.paired_aug_crop_scale_min, self.paired_aug_crop_scale_max)
            ratio = random.uniform(self.paired_aug_crop_ratio_min, self.paired_aug_crop_ratio_max)

            crop_w = int(round((area * scale * ratio) ** 0.5))
            crop_h = int(round((area * scale / max(ratio, 1e-6)) ** 0.5))
            crop_w = max(min(crop_w, width), box_w)
            crop_h = max(min(crop_h, height), box_h)

            if crop_w > width or crop_h > height:
                continue

            left_low = max(0, x_max - crop_w + 1)
            left_high = min(x_min, width - crop_w)
            top_low = max(0, y_max - crop_h + 1)
            top_high = min(y_min, height - crop_h)

            if left_low > left_high or top_low > top_high:
                continue

            left = random.randint(left_low, left_high)
            top = random.randint(top_low, top_high)
            return left, top, crop_w, crop_h

        return None

    def _random_resized_crop_triplet(self, image: Image.Image, mask: Image.Image, valid_mask: Image.Image, anomaly_present: bool):
        width, height = image.size

        if anomaly_present:
            bbox = self._mask_bbox(mask)
            crop_box = self._sample_safe_crop_box(width, height, bbox)
            if crop_box is None:
                return image, mask, valid_mask
            left, top, crop_w, crop_h = crop_box
        else:
            for _ in range(self.paired_aug_max_tries):
                scale = random.uniform(self.paired_aug_crop_scale_min, self.paired_aug_crop_scale_max)
                ratio = random.uniform(self.paired_aug_crop_ratio_min, self.paired_aug_crop_ratio_max)
                area = width * height
                crop_w = int(round((area * scale * ratio) ** 0.5))
                crop_h = int(round((area * scale / max(ratio, 1e-6)) ** 0.5))
                if 1 <= crop_w <= width and 1 <= crop_h <= height:
                    left = random.randint(0, width - crop_w)
                    top = random.randint(0, height - crop_h)
                    break
            else:
                return image, mask, valid_mask

        image = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        valid_mask = TF.resized_crop(
            valid_mask,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask, valid_mask

    def _sample_affine_params(self):
        angle = 0.0
        if random.random() < self.paired_aug_rotate_prob:
            angle = random.uniform(-self.paired_aug_rotate_deg, self.paired_aug_rotate_deg)

        translate_x = 0
        translate_y = 0
        if random.random() < self.paired_aug_translate_prob:
            max_shift = int(round(self.image_size * self.paired_aug_translate_frac))
            translate_x = random.randint(-max_shift, max_shift)
            translate_y = random.randint(-max_shift, max_shift)

        return angle, (translate_x, translate_y)

    def _apply_affine_triplet(self, image: Image.Image, mask: Image.Image, valid_mask: Image.Image, anomaly_present: bool):
        if (self.paired_aug_rotate_deg <= 0 and self.paired_aug_translate_frac <= 0):
            return image, mask, valid_mask

        image_fill = self._estimate_fill_rgb(image)
        original_fg = max(self._foreground_area(mask), 1)

        for _ in range(self.paired_aug_max_tries):
            angle, translate = self._sample_affine_params()
            if abs(angle) < 1e-6 and translate == (0, 0):
                return image, mask, valid_mask

            aug_img = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=image_fill,
            )
            aug_mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            aug_valid = TF.affine(
                valid_mask,
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

            if not anomaly_present:
                return aug_img, aug_mask, aug_valid

            kept_fg = self._foreground_area(aug_mask)
            keep_ratio = kept_fg / float(original_fg)
            if keep_ratio >= self.paired_aug_min_fg_keep_ratio:
                return aug_img, aug_mask, aug_valid

        return image, mask, valid_mask

    def _paired_spatial_augment(self, image: Image.Image, mask: Image.Image, valid_mask: Image.Image):
        anomaly_present = self._foreground_area(mask) > 0
        if (not anomaly_present) and (not self.paired_aug_apply_to_normal):
            return image, mask, valid_mask

        if random.random() > self.paired_aug_prob:
            return image, mask, valid_mask

        if random.random() < self.paired_aug_crop_prob:
            image, mask, valid_mask = self._random_resized_crop_triplet(image, mask, valid_mask, anomaly_present)

        image, mask, valid_mask = self._apply_affine_triplet(image, mask, valid_mask, anomaly_present)
        return image, mask, valid_mask

    def __getitem__(self, index):
        image_path = self.image_files[index]
        sem_path = self.sem_files[index]

        out = {"id": index}

        image = Image.open(image_path).convert("RGB")

        if sem_path is None:
            sem = Image.new("L", image.size, 0)
        else:
            sem = Image.open(sem_path).convert("L")
            if image.size != sem.size:
                raise ValueError(f"size mismatch: {image_path} vs {sem_path}")

        valid_mask = Image.new("L", image.size, 255)

        image = self._center_crop_resize(image, is_mask=False)
        sem = self._center_crop_resize(sem, is_mask=True)
        valid_mask = self._center_crop_resize(valid_mask, is_mask=True)

        if self.paired_aug_enable:
            image, sem, valid_mask = self._paired_spatial_augment(image, sem, valid_mask)

        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            sem = ImageOps.mirror(sem)
            valid_mask = ImageOps.mirror(valid_mask)

        sem_t = self.pil_to_tensor(sem)[0, :, :].float()
        sem_t = (sem_t > 127).float()

        defect_type = self._parse_defect_from_filename(image_path)
        class_id = self.DEFECT_TO_IDX.get(defect_type, 0)

        sem_idx = (sem_t * class_id).long()

        if class_id >= self.num_channels:
            raise ValueError(
                f"class_id({class_id}) >= num_channels({self.num_channels}). "
                f"请增大 num_channels，或使用 num_channels=self.NUM_DEFECT_CLASSES({self.NUM_DEFECT_CLASSES})."
            )

        H, W = sem_idx.shape
        input_label = torch.zeros(self.num_channels, H, W, dtype=torch.float32)
        sem_oh = input_label.scatter_(0, sem_idx.unsqueeze(0), 1.0)

        img_t = self.pil_to_tensor(image).float() / 255.0
        valid_t = (self.pil_to_tensor(valid_mask).float() / 255.0 > 0.5).float()
        out["image"] = (img_t - 0.5) / 0.5
        out["sem"] = sem_oh
        out["valid_mask"] = valid_t
        out["mask"] = torch.tensor(1.0, dtype=torch.float32)

        bn = os.path.basename(image_path)
        if random.random() < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping.get(bn, "")
        else:
            out["caption"] = ""

        if self.debug_aug:
            out["debug_image_path"] = image_path

        return out
