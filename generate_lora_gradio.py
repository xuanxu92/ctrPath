import gradio as gr
from PIL import Image
from api import CtrLoRA
import os

# 初始化模型
ctrlora = CtrLoRA(num_loras=1)
ctrlora.create_model(
    sd_file='/data07/shared/xxu/cvpr/may07/sd15/v1-5-pruned.ckpt',
    basecn_file='/data07/shared/xxu/cvpr/may07/ctrlora/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt',
    lora_files='/data07/shared/xxu/cvpr/may07/ctrfinetune/lightning_logs/version_2/save_path/epoch=3-step=199999_saved_lora.ckpt',
)

# 定义生成函数
def generate_image(cond_image: Image.Image, prompt: str):
    temp_path = "/home/xxu/cvpr/wacv/ctrlora/condimg/temp.png"
    cond_image.save(temp_path)

    samples = ctrlora.sample(
        cond_image_paths=temp_path,
        prompt=prompt,
        n_prompt='worst quality',
        num_samples=1,
    )
    return samples[0]

# 创建 Gradio 界面
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(label="Condition Image", type="pil"),
        gr.Textbox(label="Prompt")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="CtrLoRA Image Generator",
    description="Upload a condition image and enter a prompt to generate an image using CtrlLoRA."
)

if __name__ == "__main__":
    iface.launch(share=True)
