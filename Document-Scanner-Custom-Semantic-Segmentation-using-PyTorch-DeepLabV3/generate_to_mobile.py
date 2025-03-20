import torch
import torchvision.models.segmentation as segmentation

# 加载模型
def load_model(model_path, num_classes=2, model_name="mbv3"):
    if model_name == "mbv3":
        model = segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
    else:
        model = segmentation.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    return model

# 创建模型包装类 - 将字典输出转换为元组
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        output = self.model(x)
        # 只返回'out'键对应的值，这是主要分割结果
        return output['out']

# 转换模型
def convert_model(model_path, output_path, model_name="mbv3"):
    # 加载模型
    model = load_model(model_path, model_name=model_name)
    
    # 包装模型以获得固定结构输出
    wrapped_model = ModelWrapper(model)
    
    # 准备示例输入
    example = torch.rand(1, 3, 384, 384)
    
    # 使用trace导出模型
    traced_script_module = torch.jit.trace(wrapped_model, example)
    
    # 导出为TorchScript格式
    traced_script_module.save(output_path)
    print(f"模型已保存到: {output_path}")
    
    # 检查模型大小
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"模型大小: {size_mb:.2f} MB")

# 转换MobileNetV3模型
convert_model("model_mbv3_iou_mix_2C049.pth", "doc_scanner_mbv3.pt", "mbv3")

# 可选: 转换ResNet50模型
# convert_model("model_r50_iou_mix_2C020.pth", "doc_scanner_r50.pt", "r50")