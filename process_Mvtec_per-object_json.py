import os
import json

# ========== ==========
DATA_ROOT = "./DATA/Mvtec_per-object"  
SPLIT = "test"
IMG_DIRNAME = "Source_Images"
OUT_FILENAME = "caption.json"
VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

NORMAL_TEMPLATE = "A photo of {cls} with no defect."
# ==========================


CAPTION_BANK = {
    "bottle": {
        "broken_large": "A photo of bottle with broken large defect: a large missing chunk with jagged broken edges.",
        "broken_small": "A photo of bottle with broken small defect: a small chipped area near the rim.",
        "contamination": "A photo of bottle with contamination defect: dirty spots and foreign particles on the surface.",
    },
    "cable": {
        "bent_wire": "A photo of cable with bent wire defect: one wire is bent and deviates from the normal alignment.",
        "cut_outer_insulation": "A photo of cable with cut outer insulation defect: the outer insulating layer is sliced open.",
        "cut_inner_insulation": "A photo of cable with cut inner insulation defect: inner insulation is damaged and partially exposed.",
        "missing_wire": "A photo of cable with missing wire defect: a wire segment is missing, leaving an empty gap.",
        "poke_insulation": "A photo of cable with poke insulation defect: a puncture hole is visible on the insulation.",
        "cable_swap": "A photo of cable with cable swap defect: cable components appear swapped with incorrect arrangement.",
        "combined": "A photo of cable with combined defect: multiple defects appear simultaneously, including cuts and deformation.",
        "missing_cable": "A photo of cable with missing cable defect: a cable segment is missing, leaving a visible hole.",

    },
    "capsule": {
        "crack": "A photo of capsule with crack defect: a visible crack line across the capsule shell.",
        "faulty_imprint": "A photo of capsule with faulty imprint defect: the printed text is incomplete and blurry.",
        "squeeze": "A photo of capsule with squeeze defect: the capsule is squeezed and deformed with dents.",
        "poke": "A photo of capsule with poke defect: a small puncture mark is present on the surface.",
        "scratch": "A photo of capsule with scratch defect: visible scratch marks appear on the capsule surface.",

    },
    "carpet": {
        "cut": "A photo of carpet with cut defect: a straight cut is visible through the fabric fibers.",
        "hole": "A photo of carpet with hole defect: a hole exposes the underlying layer with missing fibers.",
        "thread": "A photo of carpet with thread defect: loose thread strands are sticking out from the carpet.",
        "metal_contamination": "A photo of carpet with metal contamination defect: tiny metallic particles are scattered on the carpet.",
        "color": "A photo of carpet with color defect: an abnormal discolored patch appears on the surface.",
    },
    "grid": {
        "broken": "A photo of grid with broken defect: parts of the grid structure are fractured and missing.",
        "glue": "A photo of grid with glue defect: glue residue forms uneven blobs on the grid.",
        "bent": "A photo of grid with bent defect: the grid lines are bent and warped.",
        "thread": "A photo of grid with thread defect: thread-like fibers are tangled on the grid.",
        "metal_contamination": "A photo of grid with metal contamination defect: small metallic particles are scattered on the grid.",

    },
    "hazelnut": {
        "crack": "A photo of hazelnut with crack defect: obvious cracks are visible on the nut shell.",
        "cut": "A photo of hazelnut with cut defect: a cut mark slices into the surface.",
        "hole": "A photo of hazelnut with hole defect: a punctured hole is visible, indicating missing material.",
        "print": "A photo of hazelnut with print defect: unusual printed marks or stains appear on the surface.",
    },
    "leather": {
        "fold": "A photo of leather with fold defect: a deep crease line is visible across the leather.",
        "cut": "A photo of leather with cut defect: a sharp cut opening appears on the surface.",
        "glue": "A photo of leather with glue defect: glue traces create glossy sticky spots.",
        "poke": "A photo of leather with poke defect: a puncture hole is visible in the leather.",
        "color": "A photo of leather with color defect: color discoloration patches appear.",
    },
    "metal_nut": {
        "scratch": "A photo of metal nut with scratch defect: abrasion lines and scratches are visible.",
        "bent": "A photo of metal nut with bent defect: the nut is deformed and slightly bent.",
        "flip": "A photo of metal nut with flip defect: the nut is flipped or oriented incorrectly.",
        "color": "A photo of metal nut with color defect: abnormal discoloration and stains appear.",
    },
    "pill": {
        "contamination": "A photo of pill with contamination defect: foreign particles and dirt spots cover the pill surface.",
        "crack": "A photo of pill with crack defect: a crack line splits part of the pill coating.",
        "faulty_imprint": "A photo of pill with faulty imprint defect: the imprint is incomplete and blurry.",
        "pill_type": "A photo of pill with pill type defect: the pill appearance differs from the expected type.",
        "combined": "A photo of pill with combined defect: multiple issues appear, including scratches and contamination.",
        "color": "A photo of pill with color defect: an abnormal discolored patch appears on the pill surface.",
        "scratch": "A photo of pill with scratch defect: multiple scratches are visible on the pill surface.",

    },
    "screw": {
        "scratch_head": "A photo of screw with scratch head defect: scratches are concentrated on the screw head.",
        "scratch_neck": "A photo of screw with scratch neck defect: abrasion marks appear around the neck region.",
        "thread_top": "A photo of screw with thread top defect: thread deformation is visible at the top.",
        "thread_side": "A photo of screw with thread side defect: side threads appear damaged and uneven.",
        "manipulated_front": "A photo of screw with manipulated front defect: the top front area is deformed and distorted.",

    },
    "tile": {
        "crack": "A photo of tile with crack defect: a long crack line runs across the tile surface.",
        "glue_strip": "A photo of tile with glue strip defect: a glue strip residue is visible as a white flake.",
        "gray_stroke": "A photo of tile with gray stroke defect: gray stroke marks appear on the tile surface.",
        "oil": "A photo of tile with oil defect: oil stain creates a glossy smear.",
        "rough": "A photo of tile with rough defect: rough patch with uneven texture is visible.",
    },
    "toothbrush": {
        "defective": "A photo of toothbrush with defective defect: bristles are deformed, missing, or irregular.",
    },
    "transistor": {
        "bent_lead": "A photo of transistor with bent lead defect: a lead pin is bent away from its normal position.",
        "cut_lead": "A photo of transistor with cut lead defect: a lead pin is cut short or missing its tip.",
        "damaged_case": "A photo of transistor with damaged case defect: the casing shows damage and broken edges.",
        "misplaced": "A photo of transistor with misplaced defect: the component appears misplaced or misaligned.",
    },
    "wood": {
        "scratch": "A photo of wood with scratch defect: clear scratch are visible along the grain.",
        "hole": "A photo of wood with hole defect: a hole with missing wood fibers appears.",
        "liquid": "A photo of wood with liquid defect: liquid stain spreads with darker patch.",
        "color": "A photo of wood with color defect: abnormal discoloration compared to surrounding wood.",
        "combined": "A photo of wood with combined defect: multiple defects appear simultaneously, including scratches and discoloration.",

    },
    "zipper": {
        "broken_teeth": "A photo of zipper with broken teeth defect: several teeth are missing or broken.",
        "split_teeth": "A photo of zipper with split teeth defect: teeth are separated and do not interlock properly.",
        "squeezed_teeth": "A photo of zipper with squeezed teeth defect: teeth are squeezed and deformed.",
        "fabric_interior": "A photo of zipper with fabric interior defect: the zipper fabric inside is damaged.",
        "fabric_border": "A photo of zipper with fabric border defect: the border fabric is torn or frayed.",
        "rough": "A photo of zipper with rough defect: rough texture and irregular surface appear.",
        "combined": "A photo of zipper with combined defect: multiple zipper defects occur together.",
    },
}


def is_image_file(fname: str) -> bool:
    return fname.lower().endswith(VALID_EXTS)


def parse_defect_from_filename(file_name: str):
    """
    - normal_001.png -> ("normal", True)
    - bent_wire_000.png -> ("bent_wire", False)
    """
    name_wo_ext, _ = os.path.splitext(file_name)
    low = name_wo_ext.lower()

    if low.startswith("normal_"):
        return "normal", True

    if "_" not in name_wo_ext:
        return None, False

    defect_raw, _id = name_wo_ext.rsplit("_", 1) 
    if not defect_raw:
        return None, False

    return defect_raw, False


def main():
    split_root = os.path.join(DATA_ROOT, SPLIT)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"找不到目录：{split_root}")

    class_names = sorted([
        d for d in os.listdir(split_root)
        if os.path.isdir(os.path.join(split_root, d))
        and not d.startswith(".")
        and d != ".ipynb_checkpoints"
    ])

    print(f"[INFO] 发现 {len(class_names)} 个类别：{class_names}")

    for cls_name in class_names:
        cls_root = os.path.join(split_root, cls_name)
        img_dir = os.path.join(cls_root, IMG_DIRNAME)
        if not os.path.isdir(img_dir):
            print(f"[WARNING] 跳过 {cls_name}，找不到目录：{img_dir}")
            continue

        if cls_name not in CAPTION_BANK:
            print(f"[WARNING] CAPTION_BANK 中没有 {cls_name} 的 caption 定义，跳过。")
            continue

        file_names = sorted(os.listdir(img_dir))
        caption_map = {}

        miss = 0
        for file_name in file_names:
            if file_name.startswith(".") or file_name == ".ipynb_checkpoints":
                continue

            full_path = os.path.join(img_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            if not is_image_file(file_name):
                continue

            defect_key, is_normal = parse_defect_from_filename(file_name)
            if defect_key is None:
                print(f"[WARNING] 文件名不符合格式，跳过：{cls_name}/{file_name}")
                continue

            if is_normal:
                caption = NORMAL_TEMPLATE.format(cls=cls_name.replace("_", " "))
                caption_map[file_name] = caption
                continue

            # defect caption
            bank = CAPTION_BANK[cls_name]
            if defect_key not in bank:
                miss += 1
                print(f"[WARNING] {cls_name}/{file_name} defect='{defect_key}' 未在 CAPTION_BANK 中定义，跳过。")
                continue

            caption_map[file_name] = bank[defect_key]

        out_path = os.path.join(cls_root, OUT_FILENAME)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(caption_map, f, ensure_ascii=False, indent=4)

        print(f"[OK] {cls_name}: 写入 {len(caption_map)} 条 -> {out_path} (missing_defect={miss})")


if __name__ == "__main__":
    main()
