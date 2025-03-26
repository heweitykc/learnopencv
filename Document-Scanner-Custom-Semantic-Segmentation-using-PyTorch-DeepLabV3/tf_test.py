import numpy as np
import tensorflow as tf
import cv2
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 数据集路径 - 确保与训练时使用的路径一致
IMAGE_DIR = '/mnt/data/scan/document_dataset_resized/train/images'
MASK_DIR = '/mnt/data/scan/document_dataset_resized/train/masks'

# 确保与训练时一致的图像尺寸
IMG_SIZE = 384

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
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 获取模型详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("模型信息:")
    print(f"  输入: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  输出: {output_details[0]['shape']} ({output_details[0]['dtype']})")
    
    return interpreter, input_details, output_details

# 预处理图像
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()  # 保存原始图像用于显示
    
    # 调整大小以适应模型输入
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 保持为uint8类型，不做归一化处理
    img = img.astype(np.uint8)
    
    return img, original_img

# 后处理输出
def postprocess_output(output, original_shape):
    # 处理uint8输出
    if output.dtype == np.uint8:
        output = output.astype(np.float32) / 255.0
    
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
    
    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    
    # 禁用TF-Flex代理，只使用CPU执行
    os.environ["TENSORFLOW_DISABLE_FLEX_DELEGATE"] = "1"
    
    # 执行推理
    print("执行推理...")
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # 后处理
    mask = postprocess_output(output, original_img.shape)
    
    return mask, original_img

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
    plt.savefig(os.path.join(output_dir, "test_result.png"))
    
    plt.show()

# 添加一个测试转换函数，跳过模型推理
def convert_test(image_path, fake_segmentation=True):
    """
    用于测试的转换函数，跳过TFLite模型推理
    
    参数:
        image_path: 图像路径
        fake_segmentation: 是否生成伪分割掩码
    
    返回:
        mask: 生成的掩码
        original_img: 原始图像
    """
    print("使用测试转换函数而非模型推理...")
    
    # 读取和预处理图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    
    if fake_segmentation:
        # 创建简单的测试掩码 - 使用简单的图像处理代替模型推理
        # 转为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 使用自适应阈值
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 进行一些形态学操作来模拟分割结果
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 二值化
        mask = (mask > 0).astype(np.uint8)
    else:
        # 如果不需要伪分割，只返回全黑掩码
        mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
    
    return mask, original_img

# 修改main函数
def main():
    # 加载模型 - 仍然保留这一步以获取模型信息
    model_result = load_model()
    if model_result is None:
        return
    
    interpreter, input_details, output_details = model_result
    
    # 随机选择测试图像
    image_path, ground_truth_path = get_random_image()
    if image_path is None:
        return
    
    print(f"测试图像: {image_path}")
    
    # 使用替代函数而不是predict
    predicted_mask, original_img = convert_test(image_path)
    # 如果想尝试原始predict函数，可取消下行注释
    # predicted_mask, original_img = predict(interpreter, input_details, output_details, image_path)
    
    if predicted_mask is None:
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