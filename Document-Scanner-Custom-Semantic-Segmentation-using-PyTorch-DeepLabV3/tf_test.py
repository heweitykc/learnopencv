import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime

# 设置TensorFlow环境变量（与训练代码一致）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少日志输出
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 动态显存分配

# 数据集路径 - 确保与训练时使用的路径一致
IMAGE_DIR = '/mnt/data/scan/document_dataset_resized/train/images'
MASK_DIR = '/mnt/data/scan/document_dataset_resized/train/masks'

# 确保与训练时一致的图像尺寸
IMG_SIZE = 416  # 与训练代码保持一致

# 设置全局字体
def set_matplotlib_chinese_font():
    try:
        # 尝试不同的中文字体，找到第一个可用的
        chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 
                         'AR PL UMing CN', 'PingFang SC', 'Hiragino Sans GB']
        
        for font_name in chinese_fonts:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path) and not font_path.endswith('DejaVuSans.ttf'):
                plt.rcParams['font.family'] = [font_name]
                print(f"已设置中文字体: {font_name}")
                return True
        
        # 如果没有找到系统字体，也可以尝试加载matplotlib自带的字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return False
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        return False

# 加载TFLite模型
def load_model(model_path="/mnt/data/scan/doc_scanner_mbv3.tflite"):
    print(f"加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        # 尝试在/mnt/data/scan/目录下查找最新的tflite文件
        data_dir = "/mnt/data/scan/"
        if os.path.exists(data_dir):
            tflite_files = glob.glob(os.path.join(data_dir, "*.tflite"))
            if tflite_files:
                latest_model = max(tflite_files, key=os.path.getctime)
                print(f"使用找到的最新模型: {latest_model}")
                model_path = latest_model
            else:
                print("未找到任何tflite模型文件")
                return None
        else:
            print(f"目录 {data_dir} 不存在")
            return None
    
    # 创建解释器实例
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 获取模型详情
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("模型信息:")
        print(f"  输入: {input_details[0]['shape']} ({input_details[0]['dtype']})")
        print(f"  输出: {output_details[0]['shape']} ({output_details[0]['dtype']})")
        
        return interpreter, input_details, output_details
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

# 预处理图像 - 保持与训练代码一致
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()  # 保存原始图像用于显示
    
    # 调整大小以适应模型输入
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 转换为float32并归一化为0-1范围
    # 训练时使用float32归一化，TFLite模型也期望接收float32
    img = img.astype(np.float32) / 255.0
    
    return img, original_img

# 后处理输出
def postprocess_output(output, original_shape):
    # 处理不同数据类型的输出
    if output.dtype == np.uint8:
        output = output.astype(np.float32) / 255.0
    elif output.dtype == np.float16:  # 处理float16输出
        output = output.astype(np.float32)
    
    # 获取掩码并调整为原始图像大小
    mask = output[0, :, :, 0]  # 获取第一个通道
    mask = (mask > 0.5).astype(np.uint8)  # 二值化
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                     interpolation=cv2.INTER_NEAREST)
    
    return mask

# 随机选择训练图像
def get_random_image():
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 图像目录 {IMAGE_DIR} 不存在")
        return None, None
    
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    if not image_paths:
        print("未找到图像文件")
        return None, None
    
    # 随机选择一张图像
    random_img_path = random.choice(image_paths)
    
    # 获取对应的掩码文件
    img_filename = os.path.basename(random_img_path)
    mask_path = os.path.join(MASK_DIR, img_filename)
    
    if not os.path.exists(mask_path):
        print(f"无法找到对应的掩码文件: {mask_path}")
        mask_path = None
    
    return random_img_path, mask_path

# 运行推理
def predict(interpreter, input_details, output_details, image_path):
    # 预处理输入
    img, original_img = preprocess_image(image_path)
    if img is None:
        return None, None
    
    # 准备输入数据
    input_tensor = np.expand_dims(img, axis=0)
    
    try:
        # 设置输入张量
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # 尝试禁用TF-Flex代理，避免类型错误
        os.environ["TENSORFLOW_DISABLE_FLEX_DELEGATE"] = "1"
        
        # 执行推理
        print("执行推理...")
        interpreter.invoke()
        
        # 获取输出
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # 后处理
        mask = postprocess_output(output, original_img.shape)
        
        return mask, original_img
    except Exception as e:
        print(f"推理失败: {e}")
        return None, None

# 可视化结果
def visualize_results(original_img, predicted_mask, ground_truth_mask=None):
    # 设置中文字体
    chinese_font_available = set_matplotlib_chinese_font()
    
    plt.figure(figsize=(15, 5))
    
    # 如果找不到中文字体，使用英文标题
    title_orig = '原始图像' if chinese_font_available else 'Original Image'
    title_pred = '预测掩码' if chinese_font_available else 'Predicted Mask'
    title_gt = '真实掩码' if chinese_font_available else 'Ground Truth'
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(title_orig)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask * 255, cmap='gray')
    plt.title(title_pred)
    plt.axis('off')
    
    if ground_truth_mask is not None:
        plt.subplot(1, 3, 3)
        # 确保ground_truth_mask是二值图像
        if ground_truth_mask.ndim > 2 and ground_truth_mask.shape[2] > 1:
            ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)
        ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)
        plt.imshow(ground_truth_mask * 255, cmap='gray')
        plt.title(title_gt)
        plt.axis('off')
        
        # 计算IoU
        intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
        union = np.logical_or(predicted_mask, ground_truth_mask).sum()
        iou = intersection / union if union > 0 else 0
        print(f"IoU分数: {iou:.4f}")
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"test_result_{timestamp}.png"))
    
    plt.show()

# 主函数
def main():
    print("\n=== 开始文档扫描分割测试 ===")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"测试分辨率: {IMG_SIZE}x{IMG_SIZE}")
    
    # 加载模型
    model_result = load_model()
    if model_result is None:
        print("模型加载失败，无法继续测试")
        return
    
    interpreter, input_details, output_details = model_result
    
    # 随机选择测试图像
    image_path, ground_truth_path = get_random_image()
    if image_path is None:
        return
    
    print(f"测试图像: {image_path}")
    
    # 执行预测
    predicted_mask, original_img = predict(interpreter, input_details, output_details, image_path)
    if predicted_mask is None or original_img is None:
        print("推理失败，无法生成预测结果")
        return
    
    # 加载真实掩码（如果有）
    ground_truth_mask = None
    if ground_truth_path:
        ground_truth_mask = cv2.imread(ground_truth_path)
    
    # 可视化结果
    visualize_results(original_img, predicted_mask, ground_truth_mask)
    
    print("测试完成!")

if __name__ == "__main__":
    main() 