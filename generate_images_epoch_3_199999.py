from api import CtrLoRA
from PIL import Image
import glob,os

import glob,os
import json
with open('/data06/shared/xxu/miccai2025/color_to_HR/tcga_text_to_image_larger/experiments/data_missing_5000.json','r') as f:
    data_5000 = json.load(f)
    
ctrlora = CtrLoRA(num_loras=1)
ctrlora.create_model(
    sd_file='/data07/shared/xxu/cvpr/may07/sd15/v1-5-pruned.ckpt',
    basecn_file='/data07/shared/xxu/cvpr/may07/ctrlora/ctrlora-basecn/ctrlora_sd15_basecn700k.ckpt',
    lora_files='/data07/shared/xxu/cvpr/may07/ctrfinetune/lightning_logs/version_2/save_path/epoch=3-step=199999_saved_lora.ckpt',
)

for i in range(len(data_5000)):
    samples = ctrlora.sample(
        cond_image_paths=os.path.join('/data06/shared/xxu/miccai2025/color_to_HR/tcga_text_to_image_larger/experiments/original_images_test_5000_inst_map_pannuke_512',data_5000[i]['path']),
        prompt=data_5000[i]['caption'],
        n_prompt='worst quality',
        num_samples=1,
    )
    save_path = os.path.join('/data07/shared/xxu/cvpr/may07/experiments/generated_images_epoch_3',data_5000[i]['path'])
    samples[0].save(save_path)