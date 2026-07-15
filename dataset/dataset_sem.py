import os
import json
import random
from pathlib import Path
from typing import Optional, List

import torch
import torch.utils.data
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
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
        if exist_in(image_file_basename,normal_files):
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
        sem = Image.open( self.sem_files[index]  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 

        assert image.size == sem.size

        
        # - - - - - center_crop, resize and random_flip - - - - - - #  

        crop_size = min(image.size)
        image = TF.center_crop(image, crop_size)
        image = image.resize( (self.image_size, self.image_size) )

        sem = TF.center_crop(sem, crop_size)
        sem = sem.resize( (self.image_size, self.image_size), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly

        if self.random_flip and random.random()<0.5:
            image = ImageOps.mirror(image)
            sem = ImageOps.mirror(sem)       

        sem = self.pil_to_tensor(sem)[0,:,:]

        if 'In' in os.path.basename(image_path):
            sem = sem/255
        elif 'Pa' in os.path.basename(image_path):
            sem = sem/255*2 
        elif 'Sc' in os.path.basename(image_path):
            sem = sem/255*3
        
        input_label = torch.zeros(152, self.image_size, self.image_size)
        sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

        out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
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

    # 单物体训练：只需要知道“该物体有哪些异常类型”
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
        obj_name: Optional[str] = None,  # 可不传：自动从路径推断
    ):
        super().__init__()
        self.image_rootdir = image_rootdir
        self.sem_rootdir = sem_rootdir
        self.caption_path = caption_path
        self.prob_use_caption = prob_use_caption
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.num_channels = int(num_channels)

        # ---------- 自动推断物体类别 ----------
        # 期望 image_rootdir 形如：.../<obj_name>/Source_Images
        if obj_name is None:
            obj_name = Path(image_rootdir).resolve().parent.name
        self.obj_name = obj_name

        if self.obj_name not in self.MVTEC_DEFECTS:
            raise ValueError(
                f"[MvtecSemanticDataset] 无法识别 obj_name={self.obj_name}。\n"
                f"请确认 image_rootdir={image_rootdir} 是否形如 .../<类别>/Source_Images，\n"
                f"或在初始化时显式传入 obj_name。"
            )

        # ---------- 单物体：defect->id 映射 ----------
        # background=0，缺陷从1开始
        self.DEFECT_TO_IDX = {"background": 0}
        for i, defect in enumerate(sorted(self.MVTEC_DEFECTS[self.obj_name]), start=1):
            self.DEFECT_TO_IDX[defect] = i
        self.NUM_DEFECT_CLASSES = 1 + len(self.MVTEC_DEFECTS[self.obj_name])

        # ---------- 读取 image / sem 文件 ----------
        image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=["png"])
        sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=["png"])
        image_files.sort()
        sem_files.sort()

        # 按 basename 配对；找不到 mask 的（例如 normal_001.png）保留图像，mask 置 None
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

        # ---------- 读取 caption ----------
        with open(caption_path, "r", encoding="utf-8") as f:
            self.image_filename_to_caption_mapping = json.load(f)

        self.pil_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.image_files)

    def total_images(self):
        return len(self)

    def _parse_defect_from_filename(self, image_path: str) -> str:
        """
        解析 defect 类型，兼容：
        - normal_001.png              -> background
        - defect_002.png              -> defect
        - obj_defect_002.png          -> defect（若前缀等于 obj_name，会自动剔除）
        """
        base = os.path.splitext(os.path.basename(image_path))[0].lower()

        # 正常样本强制视为 background
        if base.startswith("normal_"):
            return "background"

        parts = base.split("_")
        if len(parts) < 2:
            return "background"

        # 去掉最后编号（通常是数字）
        if parts[-1].isdigit():
            parts = parts[:-1]
        if len(parts) == 0:
            return "background"

        # 若第一段是类名，剔除（避免误判 defect）
        if parts[0] == self.obj_name:
            parts = parts[1:]
        if len(parts) == 0:
            return "background"

        return "_".join(parts)

    def _center_crop_resize(self, img: Image.Image, is_mask: bool) -> Image.Image:
        """
        对 image 和 mask 做一致的 center-crop + resize。
        mask 用 NEAREST，image 用 LANCZOS。
        """
        crop_size = min(img.size)
        img = TF.center_crop(img, crop_size)
        if is_mask:
            img = img.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        else:
            img = img.resize((self.image_size, self.image_size), resample=Image.LANCZOS)
        return img

    def __getitem__(self, index):
        image_path = self.image_files[index]
        sem_path = self.sem_files[index]

        out = {"id": index}

        image = Image.open(image_path).convert("RGB")

        # sem 缺失（normal）则补全 0 mask
        if sem_path is None:
            sem = Image.new("L", image.size, 0)  # 全0
        else:
            sem = Image.open(sem_path).convert("L")
            if image.size != sem.size:
                raise ValueError(f"size mismatch: {image_path} vs {sem_path}")

        # ---------- center_crop / resize ----------
        image = self._center_crop_resize(image, is_mask=False)
        sem = self._center_crop_resize(sem, is_mask=True)

        # ---------- random flip ----------
        if self.random_flip and random.random() < 0.5:
            image = ImageOps.mirror(image)
            sem = ImageOps.mirror(sem)

        # ---------- 语义编码（单物体：只区分 defect 类型） ----------
        # PIL -> tensor (H,W)，float
        sem_t = self.pil_to_tensor(sem)[0, :, :].float()

        # 二值化为 0/1（防止插值/保存导致非0/255灰度）
        sem_t = (sem_t > 127).float()

        defect_type = self._parse_defect_from_filename(image_path)
        class_id = self.DEFECT_TO_IDX.get(defect_type, 0)

        # 前景像素赋值为 class_id，背景为 0
        sem_idx = (sem_t * class_id).long()  # (H,W) in [0..K]

        # ---------- one-hot 编码 ----------
        if class_id >= self.num_channels:
            raise ValueError(
                f"class_id({class_id}) >= num_channels({self.num_channels}). "
                f"请增大 num_channels，或使用 num_channels=self.NUM_DEFECT_CLASSES({self.NUM_DEFECT_CLASSES})."
            )

        H, W = sem_idx.shape
        input_label = torch.zeros(self.num_channels, H, W, dtype=torch.float32)
        sem_oh = input_label.scatter_(0, sem_idx.unsqueeze(0), 1.0)

        # ---------- image norm ----------
        img_t = self.pil_to_tensor(image).float() / 255.0
        out["image"] = (img_t - 0.5) / 0.5
        out["sem"] = sem_oh
        out["mask"] = torch.tensor(1.0, dtype=torch.float32)

        # ---------- caption ----------
        bn = os.path.basename(image_path)
        if random.random() < self.prob_use_caption:
            out["caption"] = self.image_filename_to_caption_mapping.get(bn, "")
        else:
            out["caption"] = ""

        return out

# class MvtecSemanticDataset(torch.utils.data.Dataset):
#
#     # 单物体训练时：只需要知道“该物体有哪些异常类型”
#     MVTEC_DEFECTS = {
#         "bottle":      ["broken_large", "broken_small", "contamination"],
#         "cable":       ["bent_wire", "combined","cable_swap", "cut_inner_insulation", "cut_outer_insulation",
#                         "missing_cable", "missing_wire", "poke_insulation"],
#         "capsule":     ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
#         "carpet":      ["color", "cut", "hole", "metal_contamination", "thread"],
#         "grid":        ["bent", "broken", "glue", "metal_contamination", "thread"],
#         "hazelnut":    ["crack", "cut", "hole", "print"],
#         "leather":     ["color", "cut", "fold", "glue", "poke"],
#         "metal_nut":   ["bent", "color", "flip", "scratch"],
#         "pill":        ["color", "combined","contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
#         "screw":       ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
#         "tile":        ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
#         "toothbrush":  ["defective"],
#         "transistor":  ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
#         "wood":        ["color","combined", "hole", "liquid", "scratch"],
#         "zipper":      ["broken_teeth","combined", "fabric_border", "fabric_interior",
#                         "rough", "squeezed_teeth", "split_teeth"],
#     }
#
#     def __init__(
#         self,
#         image_rootdir,
#         sem_rootdir,
#         caption_path,
#         prob_use_caption=1,
#         image_size=512,
#         random_flip=False,
#         num_channels=152,
#         obj_name=None,  # 可不传：自动从路径推断
#     ):
#         super().__init__()
#         self.image_rootdir = image_rootdir
#         self.sem_rootdir = sem_rootdir
#         self.caption_path = caption_path
#         self.prob_use_caption = prob_use_caption
#         self.image_size = image_size
#         self.random_flip = random_flip
#         self.num_channels = num_channels
#
#         # ---------- 方案一：自动推断物体类别 ----------
#         # 期望 image_rootdir 形如：.../<obj_name>/Source_Images
#         if obj_name is None:
#             obj_name = Path(image_rootdir).resolve().parent.name
#         self.obj_name = obj_name
#
#         if self.obj_name not in self.MVTEC_DEFECTS:
#             raise ValueError(
#                 f"[MvtecSemanticDataset] 无法识别 obj_name={self.obj_name}。\n"
#                 f"请确认 image_rootdir={image_rootdir} 是否形如 .../<类别>/Source_Images，\n"
#                 f"或在初始化时显式传入 obj_name。"
#             )
#
#         # ---------- 单物体：每类自己的 defect->id 映射 ----------
#         # background=0，缺陷从1开始
#         self.DEFECT_TO_IDX = {"background": 0}
#         for i, defect in enumerate(sorted(self.MVTEC_DEFECTS[self.obj_name]), start=1):
#             self.DEFECT_TO_IDX[defect] = i
#         self.NUM_DEFECT_CLASSES = 1 + len(self.MVTEC_DEFECTS[self.obj_name])
#
#         # ---------- 读取 image / sem 文件 ----------
#         # 使用您工程已有 recursively_read
#         image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['png'])
#         sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=['png'])
#         image_files.sort()
#         sem_files.sort()
#
#         # 更稳：按 basename 配对，避免排序导致错位
#         sem_map = {os.path.basename(p): p for p in sem_files}
#         paired_images, paired_sems = [], []
#         miss = 0
#         for img_p in image_files:
#             bn = os.path.basename(img_p)
#             if bn not in sem_map:
#                 miss += 1
#                 continue
#             paired_images.append(img_p)
#             paired_sems.append(sem_map[bn])
#
#         if miss > 0:
#             print(f"[WARNING] {self.obj_name}: 有 {miss} 张图找不到对应 mask（按文件名匹配），已跳过。")
#
#         self.image_files = paired_images
#         self.sem_files = paired_sems
#
#         # ---------- 读取 caption ----------
#         with open(caption_path, 'r', encoding='utf-8') as f:
#             self.image_filename_to_caption_mapping = json.load(f)
#
#         self.pil_to_tensor = transforms.PILToTensor()
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def total_images(self):
#         return len(self)
#
#     def _parse_defect_from_filename(self, image_path: str) -> str:
#         """
#         解析 defect 类型，兼容：
#         1) defect_002.png
#         2) obj_defect_002.png  （若前缀等于 obj_name，会自动剔除）
#         """
#         base = os.path.splitext(os.path.basename(image_path))[0]  # e.g. "color_002" or "metal_nut_color_002"
#         parts = base.split("_")
#         if len(parts) < 2:
#             return "background"
#
#         # 去掉最后编号（通常是数字）
#         if parts[-1].isdigit():
#             parts = parts[:-1]
#         if len(parts) == 0:
#             return "background"
#
#         # 若第一段是类名，剔除（避免误判 defect）
#         if parts[0] == self.obj_name:
#             parts = parts[1:]
#         if len(parts) == 0:
#             return "background"
#
#         return "_".join(parts)
#
#     def __getitem__(self, index):
#         image_path = self.image_files[index]
#         sem_path = self.sem_files[index]
#
#         out = {}
#         out['id'] = index
#
#         image = Image.open(image_path).convert("RGB")
#         sem = Image.open(sem_path).convert("L")  # mask 原本为 0/255
#
#         assert image.size == sem.size, f"size mismatch: {image_path} vs {sem_path}"
#
#         # ---------- center_crop / resize / flip ----------
#         crop_size = min(image.size)
#
#         # image = TF.center_crop(image, crop_size)
#         # image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
#         #
#         # sem = TF.center_crop(sem, crop_size)
#         # sem = sem.resize((self.image_size, self.image_size), Image.NEAREST)
#
#         if self.random_flip and random.random() < 0.5:
#             image = ImageOps.mirror(image)
#             sem = ImageOps.mirror(sem)
#
#         # ---------- 语义编码（单物体：只区分 defect 类型） ----------
#         # PIL -> tensor (H,W)，float
#         sem_t = self.pil_to_tensor(sem)[0, :, :].float()
#
#         # 二值化为 0/1（防止插值/保存导致非0/255灰度）
#         sem_t = (sem_t > 127).float()
#
#         defect_type = self._parse_defect_from_filename(image_path)
#         class_id = self.DEFECT_TO_IDX.get(defect_type, 0)
#
#         # 前景像素赋值为 class_id，背景为 0
#         sem_idx = (sem_t * class_id).long()  # (H,W) in [0..K]
#
#         # ---------- one-hot 编码 ----------
#         # 默认保持 152 通道以兼容模型写死的通道数；
#         # 若您模型允许改通道，可将 num_channels 设为 self.NUM_DEFECT_CLASSES
#         if class_id >= self.num_channels:
#             raise ValueError(
#                 f"class_id({class_id}) >= num_channels({self.num_channels}). "
#                 f"请增大 num_channels，或使用 num_channels=self.NUM_DEFECT_CLASSES({self.NUM_DEFECT_CLASSES})."
#             )
#
#         input_label = torch.zeros(self.num_channels, self.image_size, self.image_size)
#         sem_oh = input_label.scatter_(0, sem_idx.unsqueeze(0), 1.0)
#
#         out['image'] = (self.pil_to_tensor(image).float() / 255.0 - 0.5) / 0.5
#         out['sem'] = sem_oh
#         out['mask'] = torch.tensor(1.0)
#
#         # ---------- caption ----------
#         bn = os.path.basename(image_path)
#         if random.random() < self.prob_use_caption:
#             out["caption"] = self.image_filename_to_caption_mapping.get(bn, "")
#         else:
#             out["caption"] = ""
#
#         return out


# # 把sem直接全部异常区分
# class MvtecSemanticDataset():
#
#     # 所有 MVTec 类别以及保留的缺陷类型（已去掉 good 和 combined）
#     MVTEC_DEFECTS = {
#         "bottle":      ["broken_large", "broken_small", "contamination"],
#         "cable":       ["bent_wire", "cable_swap", "cut_inner_insulation", "cut_outer_insulation",
#                         "missing_cable", "missing_wire", "poke_insulation"],
#         "capsule":     ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
#         "carpet":      ["color", "cut", "hole", "metal_contamination", "thread"],
#         "grid":        ["bent", "broken", "glue", "metal_contamination", "thread"],
#         "hazelnut":    ["crack", "cut", "hole", "print"],
#         "leather":     ["color", "cut", "fold", "glue", "poke"],
#         "metal_nut":   ["bent", "color", "flip", "scratch"],
#         "pill":        ["color", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
#         "screw":       ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
#         "tile":        ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
#         "toothbrush":  ["defective"],
#         "transistor":  ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
#         "wood":        ["color", "hole", "liquid", "scratch"],
#         "zipper":      ["broken_teeth", "fabric_border", "fabric_interior",
#                         "rough", "squeezed_teeth", "split_teeth"],
#     }
#
#     # 生成缺陷到类别 ID 的映射：0 留给背景
#     DEFECT_TO_IDX = {"background": 0}
#     cur_id = 1
#
#     for cls_name in sorted(MVTEC_DEFECTS.keys()):
#         for defect in sorted(MVTEC_DEFECTS[cls_name]):
#             key = f"{cls_name}_{defect}"
#             DEFECT_TO_IDX[key] = cur_id
#             cur_id += 1
#
#     NUM_DEFECT_CLASSES = cur_id  # = 背景 + 所有 (类别, 缺陷) 组合，约 70
#
#     def __init__(self, image_rootdir, sem_rootdir, caption_path, prob_use_caption=1, image_size=512, random_flip=False):
#         self.image_rootdir = image_rootdir
#         self.sem_rootdir = sem_rootdir
#         self.caption_path = caption_path
#         self.prob_use_caption = prob_use_caption
#         self.image_size = image_size
#         self.random_flip = random_flip
#
#
#         # Image and normal files
#         image_files = recursively_read(rootdir=image_rootdir, must_contain="", exts=['png'])
#         image_files.sort()
#         sem_files = recursively_read(rootdir=sem_rootdir, must_contain="", exts=['png'])
#         sem_files.sort()
#
#
#         self.image_files = image_files
#         self.sem_files = sem_files
#
#         # Open caption json
#         with open(caption_path, 'r') as f:
#             self.image_filename_to_caption_mapping = json.load(f)
#
#
#         assert len(self.image_files) == len(self.sem_files) == len(self.image_filename_to_caption_mapping)
#         self.pil_to_tensor = transforms.PILToTensor()
#
#
#     def total_images(self):
#         return len(self)
#
#
#     def __getitem__(self, index):
#
#         image_path = self.image_files[index]
#
#         out = {}
#
#         out['id'] = index
#         image = Image.open(image_path).convert("RGB")
#         sem = Image.open( self.sem_files[index]  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation
#
#         assert image.size == sem.size
#
#
#         # - - - - - center_crop, resize and random_flip - - - - - - #
#
#         crop_size = min(image.size)
#         image = TF.center_crop(image, crop_size)
#         image = image.resize( (self.image_size, self.image_size) )
#
#         sem = TF.center_crop(sem, crop_size)
#         sem = sem.resize( (self.image_size, self.image_size), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
#
#         if self.random_flip and random.random()<0.5:
#             image = ImageOps.mirror(image)
#             sem = ImageOps.mirror(sem)
#
#         # --------- MVTec 全部类别的语义编码 --------- #
#
#         # PIL -> tensor，得到 (H, W)，数值原本为 0 或 255
#         sem = self.pil_to_tensor(sem)[0, :, :].float()
#
#         # 从文件名解析出类别和缺陷名：
#         # 格式：cls_defect_编号.png 例如 "cable_bent_wire_000"
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
#         parts = base_name.split("_")
#
#         if len(parts) >= 3:
#             cls_name = parts[0]                    # e.g. "cable"
#             defect_type = "_".join(parts[1:-1])    # e.g. "bent_wire"
#             defect_key = f"{cls_name}_{defect_type}"
#         else:
#             # 不符合命名规则，直接当作背景
#             defect_key = "background"
#
#         # 二值掩码归一化为 0/1
#         sem = sem / 255.0
#
#         # 查表得到类别 ID；如果没找到，也当背景
#         class_id = self.DEFECT_TO_IDX.get(defect_key, 0)
#
#         # 前景像素赋值为该类别 ID，背景仍为 0
#         sem = sem * class_id
#
#         # 转为 long 作为类别索引图 (H, W)，每个像素是 [0, C-1]
#         sem = sem.long()
#
#         # --------- one-hot 编码 --------- #
#         # 如果你想继续保持 152 通道，可以写死为 152，只是只会用到前 NUM_DEFECT_CLASSES 个通道
#         num_channels = 152  # 或者直接用 NUM_DEFECT_CLASSES 也可以
#
#         input_label = torch.zeros(num_channels, self.image_size, self.image_size)
#         sem = input_label.scatter_(0, sem.unsqueeze(0), 1.0)
#
#         out['image'] = ( self.pil_to_tensor(image).float()/255 - 0.5 ) / 0.5
#         out['sem'] = sem
#         out['mask'] = torch.tensor(1.0)
#
#         # -------------------- caption ------------------- #
#         if random.uniform(0, 1) < self.prob_use_caption:
#             out["caption"] = self.image_filename_to_caption_mapping[os.path.basename(image_path)]
#         else:
#             out["caption"] = ""
#
#         return out

