import torch
import torchvision.models.segmentation as segmentation
import torch.quantization

# 加载TorchScript模型
def load_torchscript_model(model_path):
    # 直接使用torch.jit.load加载TorchScript模型
    model = torch.jit.load(model_path)
    model.eval()
    return model

# 加载原始模型
def load_original_model(model_path, num_classes=2, model_name="mbv3"):
    if model_name == "mbv3":
        model = segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
    else:
        model = segmentation.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()
    return model

# 量化已经是TorchScript格式的模型
def quantize_torchscript_model(model_path, output_path):
    try:
        # 加载TorchScript模型
        model = load_torchscript_model(model_path)
        print(f"已加载TorchScript模型: {model_path}")
        
        # 导出为量化版本
        # 注意：TorchScript模型已经被封装，无法直接使用标准量化API
        # 这里我们使用PyTorch Mobile优化工具
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_model = optimize_for_mobile(model)
        optimized_model.save(output_path)
        print(f"优化模型已保存到: {output_path}")
        
    except Exception as e:
        print(f"处理TorchScript模型时出错: {e}")
        print("尝试从原始权重重新创建模型并量化...")
        return False
    
    return True

# 从原始权重量化
def quantize_from_original(original_model_path, output_path, model_name="mbv3"):
    # 加载原始模型
    model = load_original_model(original_model_path, model_name=model_name)
    
    # 创建模型包装器
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            output = self.model(x)
            return output['out']
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # 配置量化设置
    wrapped_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # 准备量化
    torch.quantization.prepare(wrapped_model, inplace=True)
    
    # 校准（使用随机数据）
    with torch.no_grad():
        for _ in range(10):
            wrapped_model(torch.randn(1, 3, 384, 384))
    
    # 完成量化
    torch.quantization.convert(wrapped_model, inplace=True)
    
    # 导出模型
    example_input = torch.randn(1, 3, 384, 384)
    traced_model = torch.jit.trace(wrapped_model, example_input)
    traced_model.save(output_path)
    print(f"量化模型已保存到: {output_path}")
    
    # 检查模型大小
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"量化模型大小: {size_mb:.2f} MB")

# 主函数
def main():
    torchscript_model_path = "doc_scanner_mbv3.pt"
    original_model_path = "model_mbv3_iou_mix_2C049.pth"
    output_path = "doc_scanner_mbv3_quantized.pt"
    
    # 首先尝试直接优化TorchScript模型
    if not quantize_torchscript_model(torchscript_model_path, output_path):
        # 如果失败，从原始权重重建
        print(f"回退到从原始模型重建并量化...")
        quantize_from_original(original_model_path, output_path, "mbv3")

if __name__ == "__main__":
    main()