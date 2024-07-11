import torch
from Code.lib.depth_calibration_models import discriminator, depth_estimator
from Code.lib.DCF_ResNet_models import DCF_ResNet
from Code.lib.model import SPNet

# 载入模型
model_rgb = DCF_ResNet()
model_estimator = depth_estimator()
model = SPNet(32, 50)

model_rgb.load_state_dict(torch.load('./Checkpoint/DCF_rgb_epoch_best.pth'))
model_estimator.load_state_dict(torch.load('./Checkpoint/DCF_estimator.pth.195'))
model.load_state_dict(torch.load('./Checkpoint/CDSPNet_epoch_best.pth'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

model_rgb.eval().to(device)
model_estimator.eval().to(device)
model.eval().to(device)

# 构造随机输入图像Tensor
image = torch.randn(1, 3, 352, 352).to(device)
depth = torch.randn(1, 1, 352, 352).to(device)
x3_ = torch.randn(1, 32, 44, 44).to(device)
x4_ = torch.randn(1, 32, 22, 22).to(device)
x5_ = torch.randn(1, 32, 11, 11).to(device)
pred_depth = torch.randn(1, 1, 352, 352).to(device)

with torch.no_grad():
    torch.onnx.export(
        model_rgb,
        image,
        "DCF_rgb_epoch_best.onnx",
        opset_version=11,
        input_names=['image'],
        output_names=['_', 'res_r', 'x3_r', 'x4_r', 'x5_r']
    )

with torch.no_grad():
    torch.onnx.export(
        model_estimator,
        (image, x3_, x4_, x5_),
        "DCF_estimator_195.onnx",
        opset_version=11,
        input_names=['image', 'x3_', 'x4_', 'x5_'],
        output_names=['pred_depth']
    )

with torch.no_grad():
    torch.onnx.export(
        model,
        (image, depth, pred_depth),
        "CDSPNet_epoch_best.onnx",
        opset_version=11,
        input_names=['image', 'depth', 'pred_depth'],
        output_names=['pre_res']
    )