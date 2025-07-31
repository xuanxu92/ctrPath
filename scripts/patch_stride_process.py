import os
import json
import cv2
from tqdm import tqdm

# 输入 jsonl 路径
jsonl_path = "/data07/shared/xxu/cvpr/may07/pathgen/prompt.jsonl"

# 输出目录
output_img_dir = "/data07/shared/xxu/cvpr/may07/data_2k/patches_from_jsonl/images_512"
output_cond_dir = "/data07/shared/xxu/cvpr/may07/data_2k/patches_from_jsonl/nuclei_512"
output_prompt_dir = "/data07/shared/xxu/cvpr/may07/data_2k/patches_from_jsonl/prompts"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_cond_dir, exist_ok=True)
os.makedirs(output_prompt_dir, exist_ok=True)

# patch 参数
patch_size = 512
stride = 256

def extract_patches(img, patch_size, stride):
    H, W = img.shape[:2]
    patches, coords = [], []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            coords.append((i, j))
    return patches, coords

# 读取 jsonl 并处理
with open(jsonl_path, "r") as f:
    for line in tqdm(f, desc="Processing JSONL items"):
        item = json.loads(line.strip())
        source_path = item["source"].replace('_512/','/')
        target_path = item["target"]
        prompt = item["prompt"]

        base_name = os.path.splitext(os.path.basename(source_path))[0]  # 用 source 命名
        if os.path.exists(target_path) and os.path.exists(source_path):
            img = cv2.imread(target_path)
            cond = cv2.imread(source_path)

            if img is None or cond is None:
                print(f"Warning: Cannot read image {target_path} or {source_path}")
                continue

            img_patches, coords = extract_patches(img, patch_size, stride)
            cond_patches, _ = extract_patches(cond, patch_size, stride)

            for (i, j), img_patch, cond_patch in zip(coords, img_patches, cond_patches):
                patch_id = f"{base_name}_r{i}_c{j}"
                img_path_out = os.path.join(output_img_dir, f"{patch_id}.png")
                cond_path_out = os.path.join(output_cond_dir, f"{patch_id}.png")
                prompt_path_out = os.path.join(output_prompt_dir, f"{patch_id}.txt")

                cv2.imwrite(img_path_out, img_patch)
                cv2.imwrite(cond_path_out, cond_patch)
                with open(prompt_path_out, "w") as pf:
                    pf.write(prompt)
        else:
            print('not found')