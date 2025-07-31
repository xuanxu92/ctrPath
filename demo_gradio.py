import gradio as gr
from PIL import Image
from api import CtrLoRA
import tempfile
import os
import torch
import pyiqa
from utils import util_image
import time

# 初始化模型
ctrlora = CtrLoRA(num_loras=1)
ctrlora.create_model(
    sd_file='/data07/shared/xxu/cvpr/may07/sd15/v1-5-pruned.ckpt',
    basecn_file='/data07/shared/xxu/cvpr/may07/ctrlora/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt',
    lora_files='/data07/shared/xxu/cvpr/may07/ctrfinetune/lightning_logs/version_2/save_path/epoch=3-step=199999_saved_lora.ckpt',
)

# 初始化 IQA 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_dict = {
    "CLIP-IQA": pyiqa.create_metric('clipiqa').to(device),
    "MUSIQ": pyiqa.create_metric('musiq').to(device),
    "NIQE": pyiqa.create_metric('niqe').to(device),
    "BRISQUE": pyiqa.create_metric('brisque').to(device),
    "NRQM": pyiqa.create_metric('nrqm').to(device),
}
metric_paired_dict = {
    "PSNR": pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
    "SSIM": pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
    "LPIPS": pyiqa.create_metric('lpips').to(device),
    "ST-LPIPS": pyiqa.create_metric('stlpips').to(device),
}

def evaluate_image(gen_img_path, gt_img_path=None):
    results = {}
    try:
        im_in = util_image.imread(gen_img_path, chn='rgb', dtype='float32')
        im_in_tensor = util_image.img2tensor(im_in).to(device)

        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                results[key] = metric(im_in_tensor).item()

        if gt_img_path:
            im_ref = util_image.imread(gt_img_path, chn='rgb', dtype='float32')
            im_ref_tensor = util_image.img2tensor(im_ref).to(device)
            for key, metric in metric_paired_dict.items():
                with torch.cuda.amp.autocast():
                    results[key] = metric(im_in_tensor, im_ref_tensor).item()
    except Exception as e:
        results["Error"] = str(e)

    return results

def generate_and_evaluate(cond_image: Image.Image, prompt: str, gt_image: Image.Image = None):
    # 创建输出目录
    output_dir = "/home/xxu/cvpr/wacv/ctrlora/demo_output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 保存条件图像
    cond_image = cond_image.convert("RGB")
    cond_path = os.path.join(output_dir, f"{timestamp}_cond.png")
    cond_image.save(cond_path)

    # 生成图像
    samples = ctrlora.sample(
        cond_image_paths=cond_path,
        prompt=prompt,
        n_prompt='worst quality',
        num_samples=1,
    )
    gen_img = samples[0].convert("RGB")
    gen_img_path = os.path.join(output_dir, f"{timestamp}_gen.png")
    gen_img.save(gen_img_path)

    # 保存 GT 图像（如果提供）
    gt_img_path = None
    if gt_image:
        gt_image = gt_image.convert("RGB")
        gt_img_path = os.path.join(output_dir, f"{timestamp}_gt.png")
        gt_image.save(gt_img_path)

    # 计算评估指标
    results = evaluate_image(gen_img_path, gt_img_path)
    results_str = "\n".join(f"{k}: {v:.4f}" for k, v in results.items())

    return gen_img, results_str

iface = gr.Interface(
    fn=generate_and_evaluate,
    inputs=[
        gr.Image(label="Condition Image", type="pil"),
        gr.Textbox(label="Prompt"),
        gr.Image(label="Ground Truth Image (Optional, for evaluation)", type="pil"),
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Evaluation Metrics"),
    ],
    title="CtrLoRA Image Generator with Evaluation",
    description="Upload a condition image and enter a prompt. Optionally upload a ground truth image to compute full-reference metrics like PSNR, SSIM, LPIPS."
)

if __name__ == "__main__":
    iface.launch(share=True)
