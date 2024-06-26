import torch
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

deploy_cfg = "../mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py"
model_cfg = "cascade_rcnn_swin-t-p4-w7_fpn_1x_algae_demo_dist.py"
device = "cuda"
backend_model = ["mmdeploy_models/onnx_algae/model.onnx"]
image = "20210315_pc_19_0m_D1.jpg"

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# # build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

print(result)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name="visualize",
    output_file="onnx_detected_20210315_pc_19_0m_D1.jpg",
)
