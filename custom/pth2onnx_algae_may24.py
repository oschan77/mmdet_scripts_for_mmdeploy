from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = "20210315_pc_19_0m_D1.jpg"
work_dir = "mmdeploy_models/onnx_algae_may24"
save_file = "model_may24.onnx"
deploy_cfg = "../mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"
model_cfg = "cascade_rcnn_swin-t-p4-w7_fpn_1x_algae_dist_may24.py"
model_checkpoint = "may24_epoch_500.pth"
device = "cuda"

# convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg, model_checkpoint, device)

# extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)
