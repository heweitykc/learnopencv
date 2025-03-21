import numpy as np
import tensorflow as tf
import cv2

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path="doc_scanner_mbv3.tflite")
interpreter.allocate_tensors()

# 获取输入输出细节
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 预处理图像
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img = img.astype(np.uint8)  # 确保是uint8类型
    return img

# 运行推理
def predict(image_path):
    # 预处理输入
    img = preprocess_image(image_path)
    input_tensor = np.expand_dims(img, axis=0)
    
    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    
    # 执行推理
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # 后处理（将uint8输出转为概率图）
    if output.dtype == np.uint8:
        output = output.astype(np.float32) / 255.0
    
    # 二值化掩码
    mask = (output[0] > 0.5).astype(np.uint8)
    
    return mask

# 使用示例
mask = predict("test_image.jpg")
cv2.imwrite("output_mask.png", mask * 255)