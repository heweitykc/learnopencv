# -*- coding: utf-8 -*-
"""Custom_training_deeplabv3_mbv3_TFLite.ipynb"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# 避免内存溢出错误
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# 设置参数
IMG_SIZE = 384
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 2  # 背景和文档

# 假设你的数据集结构：
# - images/ 包含所有图像
# - masks/ 包含所有掩码 (二值化: 0=背景, 1=文档)

# 1. 数据准备
def get_dataset_paths(images_dir='images', masks_dir='masks'):
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
    
    # 确保图像和掩码数量一致
    assert len(image_paths) == len(mask_paths), "图像和掩码数量不匹配!"
    
    return image_paths, mask_paths

# 数据增强
def data_augmentation(image, mask):
    # 50%几率应用增强
    if tf.random.uniform(()) > 0.5:
        # 随机翻转
        image = tf.image.random_flip_left_right(image)
        mask = tf.image.random_flip_left_right(mask)
        
    # 随机亮度、对比度和饱和度
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    
    # 保持值在[0,1]范围内
    image = tf.clip_by_value(image, 0, 1)
    
    return image, mask

# 解析图像和掩码
def parse_image(img_path, mask_path):
    # 读取图像
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    # 读取掩码
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
    mask = tf.cast(mask > 0, tf.float32)  # 二值化掩码
    
    return image, mask

# 创建TensorFlow数据集
def create_dataset(image_paths, mask_paths, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# 2. 创建DeepLabV3+模型
def DeepLabV3Plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # 基础模型 - MobileNetV3Large
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 冻结早期层
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # 获取特征图
    input_image = base_model.input
    
    # 低级特征 (浅层特征)
    low_level_features = base_model.get_layer('expanded_conv_project_BN').output
    low_level_features = tf.keras.layers.Conv2D(48, 1, padding='same', 
                                              use_bias=False)(low_level_features)
    low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
    low_level_features = tf.keras.layers.Activation('relu')(low_level_features)
    
    # ASPP (Atrous Spatial Pyramid Pooling)
    x = base_model.output
    
    # 1x1卷积
    aspp_conv1 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    aspp_conv1 = tf.keras.layers.BatchNormalization()(aspp_conv1)
    aspp_conv1 = tf.keras.layers.Activation('relu')(aspp_conv1)
    
    # 空洞卷积率=6
    aspp_conv2 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                      dilation_rate=6, use_bias=False)(x)
    aspp_conv2 = tf.keras.layers.BatchNormalization()(aspp_conv2)
    aspp_conv2 = tf.keras.layers.Activation('relu')(aspp_conv2)
    
    # 空洞卷积率=12
    aspp_conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                      dilation_rate=12, use_bias=False)(x)
    aspp_conv3 = tf.keras.layers.BatchNormalization()(aspp_conv3)
    aspp_conv3 = tf.keras.layers.Activation('relu')(aspp_conv3)
    
    # 空洞卷积率=18
    aspp_conv4 = tf.keras.layers.Conv2D(256, 3, padding='same', 
                                      dilation_rate=18, use_bias=False)(x)
    aspp_conv4 = tf.keras.layers.BatchNormalization()(aspp_conv4)
    aspp_conv4 = tf.keras.layers.Activation('relu')(aspp_conv4)
    
    # 全局平均池化
    aspp_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    aspp_pool = tf.keras.layers.Reshape((1, 1, -1))(aspp_pool)
    aspp_pool = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(aspp_pool)
    aspp_pool = tf.keras.layers.BatchNormalization()(aspp_pool)
    aspp_pool = tf.keras.layers.Activation('relu')(aspp_pool)
    aspp_pool = tf.keras.layers.UpSampling2D(size=(12, 12), interpolation='bilinear')(aspp_pool)
    
    # 合并ASPP分支
    aspp_concat = tf.keras.layers.Concatenate()([aspp_conv1, aspp_conv2, 
                                                aspp_conv3, aspp_conv4, aspp_pool])
    aspp_concat = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(aspp_concat)
    aspp_concat = tf.keras.layers.BatchNormalization()(aspp_concat)
    aspp_concat = tf.keras.layers.Activation('relu')(aspp_concat)
    aspp_concat = tf.keras.layers.Dropout(0.5)(aspp_concat)
    
    # 上采样ASPP特征
    aspp_upsampled = tf.keras.layers.UpSampling2D(size=(4, 4), 
                                                interpolation='bilinear')(aspp_concat)
    
    # 合并低级特征和ASPP特征
    decoder_concat = tf.keras.layers.Concatenate()([aspp_upsampled, low_level_features])
    
    # 解码器
    decoder = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_concat)
    decoder = tf.keras.layers.Dropout(0.5)(decoder)
    decoder = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(decoder)
    decoder = tf.keras.layers.Dropout(0.1)(decoder)
    
    # 最终上采样和预测
    decoder = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(decoder)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same')(decoder)
    
    # 创建模型
    model = tf.keras.Model(inputs=input_image, outputs=outputs)
    return model

# 3. 定义损失函数和评估指标
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.sigmoid(y_pred))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(tf.keras.backend.sigmoid(y_pred), epsilon, 1. - epsilon)
    
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    focal_loss = - alpha_t * tf.pow(1. - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_focal_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(tf.cast(tf.keras.backend.sigmoid(y_pred) > 0.5, tf.float32))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 4. 主函数：训练和转换模型
def train_and_convert():
    # 获取数据路径
    image_paths, mask_paths = get_dataset_paths()
    
    # 分割训练和验证集
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42)
    
    print(f"训练图像数量: {len(train_img_paths)}")
    print(f"验证图像数量: {len(val_img_paths)}")
    
    # 创建数据集
    train_dataset = create_dataset(train_img_paths, train_mask_paths, training=True)
    val_dataset = create_dataset(val_img_paths, val_mask_paths, training=False)
    
    # 创建模型
    model = DeepLabV3Plus()
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coef, iou_metric]
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            'deeplabv3plus_mbv3_best.h5',
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # 绘制训练历史
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['dice_coef'], label='Train Dice')
    plt.plot(history.history['val_dice_coef'], label='Val Dice')
    plt.legend()
    plt.title('Dice Coefficient')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['iou_metric'], label='Train IoU')
    plt.plot(history.history['val_iou_metric'], label='Val IoU')
    plt.legend()
    plt.title('IoU')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # 保存模型
    model.save('deeplabv3plus_mbv3_final.h5')
    print("模型训练完成并保存")
    
    # 转换为TensorFlow Lite
    # 加载最佳模型
    model = tf.keras.models.load_model(
        'deeplabv3plus_mbv3_best.h5',
        custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'binary_focal_loss': binary_focal_loss,
            'combined_loss': combined_loss,
            'iou_metric': iou_metric
        }
    )
    
    # 优化模型结构用于推断
    def preprocess_input(x):
        return x

    def post_processing(x):
        return tf.sigmoid(x)
    
    # 创建一个推断模型，包含sigmoid激活函数
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocessed = tf.keras.layers.Lambda(preprocess_input)(input_tensor)
    features = model(preprocessed)
    outputs = tf.keras.layers.Lambda(post_processing)(features)
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
    
    # TFLite转换
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # 优化设置
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 用于量化校准的代表性数据集
    def representative_dataset():
        for image_batch, _ in train_dataset.take(100):
            for image in image_batch:
                image = tf.expand_dims(image, axis=0)
                yield [image]
    
    # 应用完整整数量化
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # 转换模型
    tflite_model = converter.convert()
    
    # 保存TFLite模型
    with open('doc_scanner_mbv3.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite模型已保存")
    
    # 打印模型大小
    print(f"TFLite模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    # 测试TFLite模型
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # 获取输入输出细节
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("TFLite模型输入详情:", input_details)
    print("TFLite模型输出详情:", output_details)
    
    # 测试一个样本
    sample_image, sample_mask = next(iter(val_dataset))[0][0], next(iter(val_dataset))[1][0]
    
    # 转换为uint8格式
    sample_image_uint8 = tf.cast(sample_image * 255, tf.uint8).numpy()
    
    # 设置输入
    interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(sample_image_uint8, 0))
    
    # 推理
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.astype(np.float32) / 255.0  # 从uint8转回float32
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sample_mask[:,:,0], cmap='gray')
    plt.title('真实掩码')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output[0,:,:,0], cmap='gray')
    plt.title('TFLite预测')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('tflite_test_result.png')
    plt.show()

# 运行主函数
if __name__ == "__main__":
    train_and_convert()