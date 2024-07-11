"""
    main_version_3
    在main_version_2的基础上，优化了GUI界面：
        1）删除了pred_depth组件，pred_res组件和gt组件，将生成结果直接输出在output_pred(Gallery)组件中
        2）调整了第一行的GUI布局，将比例调整为3：3：4，同时将rgb_image组件和depth组件的min_width调整为70
        3）添加了ClearAllButton组件，可以一键清除output_pred(Gallery)组件在内的所有图像
        4）启动服务器前，定义随机张量在onnxruntime-gup上推理，保证启动后onnx模型推理时间正常（解决了服务器启动后第一次推理时间过长的问题）
"""

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime
import os
from PIL import Image
from data_gradio_version_2 import test_dataset


""" 确保服务器在本地启动 """
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"


""" 载入onnx模型，获取ONNX Runtime推理器 """
model_rgb_ort_session = onnxruntime.InferenceSession("DCF_rgb_epoch_best.onnx", providers=["CUDAExecutionProvider"])
model_estimator_session = onnxruntime.InferenceSession("DCF_estimator_195.onnx", providers=["CUDAExecutionProvider"])
model_ort_session = onnxruntime.InferenceSession("CDSPNet_epoch_best.onnx", providers=["CUDAExecutionProvider"])


""" 清除函数 """
def cle():
    # return None, None, None, None, None
    return None, None

def cleAll():
    return None, None, None


""" 查询folder_path中名为image_name的图片 """
def find_image_by_name(image_name, folder_path):
    for root, dirs, files in os.walk(folder_path):
        if image_name in files:
            image_path = os.path.join(root, image_name)
            return image_path
    return None


""" 获取GT图片路径 """
def getGTPath(image_path, depth_path):
    name = depth_path.split('\\')[-1]
    return find_image_by_name(name, "./Data/TestDataset2_GT")


"""启动服务器前的第一次推理"""
def first_interface(iamge, depth):
    # 推理model_rgb的onnx模型
    _, res_r, x3_r, x4_r, x5_r = model_rgb_ort_session.run(None, {'image': image})
    x3_r, x4_r, x5_r = torch.tensor(x3_r), torch.tensor(x4_r), torch.tensor(x5_r)
    x3_, x4_, x5_ = x3_r.detach().numpy().astype(np.float32), x4_r.detach().numpy().astype(
        np.float32), x5_r.detach().numpy().astype(np.float32)
    # 推理model_estimator的onnx模型
    pred_depth = model_estimator_session.run(None, {'image': image, 'x3_': x3_, 'x4_': x4_, 'x5_': x5_})
    pred_depth = torch.tensor(pred_depth[0])
    temp = pred_depth
    pred_depth = pred_depth.detach().numpy().astype(np.float32)
    # 推理model的onnx模型
    output = model_ort_session.run(None, {'image': image, 'depth': depth, 'pred_depth': pred_depth})
    res = torch.tensor(output[2])
    return True

""" 推理函数 """
def cal(image_path, depth_path, output_pred, testsize, device):
    testsize = 352
    gt_path = getGTPath(image_path, depth_path)
    # 读取image_path, depth_path, gt_path的数据，并做预处理，转换为tensor格式
    image_pil, depth_pil, gt_pil, image, gt, depth = test_dataset(image_path, gt_path, depth_path, testsize).load_data()
    # 将image, depth, gt_shape转换为numpy格式
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.detach().numpy().astype(np.float32)
    depth = depth.detach().numpy().astype(np.float32)
    # 推理model_rgb的onnx模型
    _, res_r, x3_r, x4_r, x5_r = model_rgb_ort_session.run(None, {'image': image})
    x3_r, x4_r, x5_r = torch.tensor(x3_r), torch.tensor(x4_r), torch.tensor(x5_r)
    x3_, x4_, x5_ = x3_r.detach().numpy().astype(np.float32), x4_r.detach().numpy().astype(
        np.float32), x5_r.detach().numpy().astype(np.float32)
    # 推理model_estimator的onnx模型
    pred_depth = model_estimator_session.run(None, {'image': image, 'x3_': x3_, 'x4_': x4_, 'x5_': x5_})
    pred_depth = torch.tensor(pred_depth[0])
    temp = pred_depth
    pred_depth = pred_depth.detach().numpy().astype(np.float32)
    # 推理model的onnx模型
    output = model_ort_session.run(None, {'image': image, 'depth': depth, 'pred_depth': pred_depth})
    res = torch.tensor(output[2])
    # 将res激活并保存为pil格式图片
    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = Image.fromarray(res * 255)
    # 将推理得到的pre_depth(temp)保存为pil格式
    temp = temp.data.cpu().numpy().squeeze()
    temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)
    pred_depth = Image.fromarray(temp * 255)
    # 将image, depth, pred_depth, pred_res, gt保存在Gallery(output_pred)中
    if output_pred == None:
        output_pred = [(image_pil, "image"), (depth_pil, "depth"), (pred_depth, "pred_depth"), (res, "pred_res"),
                       (gt_pil, "GT")]
    else:
        output_pred = output_pred + [(image_pil, "image"), (depth_pil, "depth"), (pred_depth, "pred_depth"),
                                     (res, "pred_res"), (gt_pil, "GT")]
    # 将numpy格式的pred_depth和pred_res返回
    # return [pred_depth, res, gt_pil, output_pred]
    return output_pred


title = 'RGB-D显著目标识别演示系统'
description = '输入RGB图像和深度图，点击Generate按钮，生成显著目标的掩码'

""" Gradio框架的GUI界面 """
with gr.Blocks(title=title) as demo:
    # gr.Markdown("""<div align=center style="font-size:30px">RGB-D显著目标识别演示系统</div>""")
    gr.Markdown("""# RGB-D显著目标识别演示系统""")
    with gr.Tab(label="CDSPNet"):
        with gr.Row():
            with gr.Column(scale=3, min_width=70):
                rgb_image = gr.Image(label="image", sources=["upload"], show_share_button=True,
                                     type='filepath')
            with gr.Column(scale=3, min_width=70):
                depth = gr.Image(label="depth", sources=["upload"], show_share_button=True,
                                 type='filepath')
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column():
                        testsize = gr.Slider(label="Test Size", minimum=256, maximum=512, value=352, interactive=False)
                        device = gr.Radio(choices=["GPU", "CPU"], label="Device", value="GPU")
                with gr.Row():
                    with gr.Column(scale=1, min_width=50):
                        # clearButton 仅清除rgb_image组件和depth组件
                        clearButton = gr.Button(value="Clear", size="sm", variant="secondary", interactive=True)
                    with gr.Column(scale=1, min_width=50):
                        # clearAllButton 清楚rgb_image组件，depth组件和output_pred组件
                        clearAllButton = gr.Button(value="ClearAll", size="sm", variant="secondary", interactive=True)
                with gr.Row():
                    generateButton = gr.Button(value="Generate", size="sm", variant="primary", interactive=True)
        # with gr.Row():
        #     with gr.Column(scale=1):
        #         pred_depth = gr.Image(label="pred_image", type='pil', interactive=False, visible=False)
        #     with gr.Column(scale=1):
        #         pred_res = gr.Image(label="pred_res", type='pil', interactive=False, visible=False)
        #     with gr.Column(scale=1):
        #         gt = gr.Image(label="GT", type='pil', interactive=False, visible=False)
        with gr.Row():
            output_pred = gr.Gallery(columns=5, label="output", interactive=False, type='pil', object_fit="contain",
                                     height="auto")
        # clearButton.click(fn=cle, inputs=None, outputs=[rgb_image, depth, pred_depth, pred_res, gt])
        # generateButton.click(fn=cal, inputs=[rgb_image, depth, output_pred, testsize, device],
        #                      outputs=[pred_depth, pred_res, gt, output_pred])
        clearButton.click(fn=cle, inputs=None, outputs=[rgb_image, depth])
        clearAllButton.click(fn=cleAll, inputs=None, outputs=[rgb_image, depth, output_pred])
        generateButton.click(fn=cal, inputs=[rgb_image, depth, output_pred, testsize, device], outputs=output_pred)

if __name__ == "__main__":
    image = torch.randn(1, 3, 352, 352).detach().numpy().astype(np.float32)
    depth = torch.randn(1, 1, 352, 352).detach().numpy().astype(np.float32)
    if first_interface(image, depth):
        print("第一次推理成功！可以启动demo服务器！")
    demo.launch()  # 启动服务器
