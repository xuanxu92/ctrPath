# 🧬 CtrPath: Controllable Histopathology Image Generation via Conditional Diffusion

CtrPath is an open-source Python tool designed for controllable histopathology image generation using diffusion models. It leverages a lightweight conditional architecture based on [CtrlLoRA](https://github.com/xyfJASON/ctrlora), and supports structural guidance via nuclei instance maps and semantic control through text prompts.

## ✨ Key Features

- Generate 512×512 histopathology image patches
- Conditioned on nuclei instance maps and pathology-style text prompts
- Optional ground truth upload for computing full-reference metrics
- No-reference metrics supported out of the box (CLIP-IQA, NIQE, etc.)
- Interactive Gradio interface

---

## 📦 Installation

First, install dependencies for [CtrlLoRA](https://github.com/xyfJASON/ctrlora):

```bash
git clone https://github.com/xyfJASON/ctrlora.git
cd ctrlora
conda env create -f environment.yaml
conda activate ctrldm
```

Then install this repo:

```bash
git clone https://github.com/xuanxu92/ctrPath.git
cd ctrPath
pip install -r requirements.txt
```

---

## 📥 Checkpoint Setup

Download the following checkpoints:

| Component             | Checkpoint Filename                          |
|-----------------------|-----------------------------------------------|
| Stable Diffusion v1.5 | `v1-5-pruned.ckpt`                            |
| CtrlLoRA BaseCN       | `ctrlora_sd15_basecn700k.ckpt`               |
| CtrPath LoRA Weights  | `epoch=3-step=199999_saved_lora.ckpt`        |

> ☁️ [Download Checkpoints from Google Drive](#) (link to be added)

Place the checkpoints in a known directory and update paths in `demo.py`.

---

## 🚀 Run the Demo

Launch the interactive demo with:

```bash
python demo.py
```

### Inputs

- 🖼️ **Condition Image**: Nuclei mask (PNG)
- 🧠 **Prompt**: e.g., "a low-grade tumor with necrosis"
- 🧪 **Ground Truth Image** (Optional): Reference for full metrics

### Outputs

- Generated pathology image
- Evaluation metrics (PSNR, SSIM, LPIPS, CLIP-IQA, etc.)

---

## 📊 Evaluation Metrics

- **No-reference**: CLIP-IQA, NIQE, BRISQUE, MUSIQ, NRQM
- **Full-reference** (if GT provided): PSNR, SSIM, LPIPS, ST-LPIPS

---

## 🧑‍💻 Acknowledgments

Built on [CtrlLoRA](https://github.com/xyfJASON/ctrlora), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and [pyiqa](https://github.com/chaofengc/IQA-PyTorch).

## 📬 Contact

Questions or feedback: xuanxu92@gmail.com

