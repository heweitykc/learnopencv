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
import gc

TRAIN_LEN = 0
VAL_LEN = 0
EPOCHS = 50
NUM_CLASSES = 2

# V100 * 1, 32GB显存, 12核CPU, 92GB内存优化参数
IMG_SIZE = 384  # 保持原分辨率，足够文档边缘检测
BATCH_SIZE = 32  # 减小批量缓解内存压力

# 发布参数
# TRAIN_LEN = 0
# VAL_LEN = 0
# IMG_SIZE = 384
# BATCH_SIZE = 8
# EPOCHS = 50
# NUM_CLASSES = 2  # 背景和文档



train_dataset = None
val_dataset = None


# 1. 数据准备, 图片尺寸大概384*500左右
def get_dataset_paths(images_dir='/mnt/data/scan/document_dataset_resized/train/images', masks_dir='/mnt/data/scan/document_dataset_resized/train/masks'):
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))

    # 限制返回的数据集大小为指定的训练和验证长度
    if TRAIN_LEN > 0 and len(image_paths) > TRAIN_LEN + VAL_LEN:
        image_paths = image_paths[:TRAIN_LEN + VAL_LEN]
        mask_paths = mask_paths[:TRAIN_LEN + VAL_LEN]

    print(len(image_paths))
    print(len(mask_paths))
    # 确保图像和掩码数量一致
    assert len(image_paths) == len(mask_paths), "图像和掩码数量不匹配!"

    return image_paths, mask_paths

# 数据增强
@tf.function
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
@tf.function
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

# 优化数据加载函数，减轻CPU负担
def create_dataset(image_paths, mask_paths, training=True):
    AUTOTUNE = tf.data.AUTOTUNE
    
    # 设置options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    
    # 优化点1: 加载前先处理解码数据，减少重复动作
    if len(image_paths) > 1000 and training:
        # 对于大数据集，只预加载部分数据
        image_paths = image_paths[:1000]
        mask_paths = mask_paths[:1000]
    
    # 将路径转换为数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.with_options(options)
    
    # 优化点2: 减少并行处理线程，避免CPU过载
    if training:
        # 减小shuffle buffer进一步减轻内存和CPU压力
        dataset = dataset.shuffle(buffer_size=500)  # 进一步减小
        dataset = dataset.map(parse_image, num_parallel_calls=4)  # 从6减到4
        # 优化点3: 减少数据增强复杂度，提高速度
        dataset = dataset.map(simple_data_augmentation, num_parallel_calls=4)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE)
    else:
        dataset = dataset.map(parse_image, num_parallel_calls=4)
        dataset = dataset.batch(BATCH_SIZE)
    
    # 使用较小的预取值
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)  # AUTOTUNE让TF自行决定最优预取值
    return dataset

# 优化点4: 简化数据增强函数，减轻CPU负担
@tf.function
def simple_data_augmentation(image, mask):
    # 简化增强操作，只保留最重要的部分
    if tf.random.uniform(()) > 0.5:
        # 只做水平翻转
        image = tf.image.random_flip_left_right(image)
        mask = tf.image.random_flip_left_right(mask)
    
    # 移除亮度、对比度等重复处理
    return image, mask

# 2. 创建DeepLabV3+模型
def DeepLabV3Plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # 基础模型 - MobileNetV3Large
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights=None,  # 先不加载预训练权重
        minimalistic=True
    )
    
    # 加载预训练权重到基础模型
    temp_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        minimalistic=True
    )
    
    # 复制可以共享的权重
    for layer, temp_layer in zip(base_model.layers, temp_model.layers):
        if layer.name == temp_layer.name:  # 确保层名称匹配
            try:
                layer.set_weights(temp_layer.get_weights())
            except ValueError:
                print(f"跳过层 {layer.name} 的权重加载")
    
    # 冻结早期层
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # 获取特征图
    input_image = base_model.input

    # 修改这里：使用正确的层名称 (expanded_conv_3_project 而不是 expanded_conv_3/project)
    low_level_features = base_model.get_layer('expanded_conv_3_project').output
    low_level_features = tf.keras.layers.Conv2D(48, 1, padding='same',
                                              use_bias=False)(low_level_features)
    low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
    low_level_features = tf.keras.layers.Activation('relu')(low_level_features)

    # ASPP (Atrous Spatial Pyramid Pooling)
    x = base_model.output

    # 获取特征图尺寸 - 注意这里只获取一次确保一致性
    x_shape = tf.keras.backend.int_shape(x)
    
    # 1x1卷积
    aspp_conv1 = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    aspp_conv1 = tf.keras.layers.BatchNormalization()(aspp_conv1)
    aspp_conv1 = tf.keras.layers.Activation('relu')(aspp_conv1)

    # 空洞卷积率=6
    aspp_conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    aspp_conv2 = tf.keras.layers.BatchNormalization()(aspp_conv2)
    aspp_conv2 = tf.keras.layers.Activation('relu')(aspp_conv2)

    # 空洞卷积率=12
    aspp_conv3 = tf.keras.layers.Conv2D(128, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    aspp_conv3 = tf.keras.layers.BatchNormalization()(aspp_conv3)
    aspp_conv3 = tf.keras.layers.Activation('relu')(aspp_conv3)

    # 全局平均池化 - 修改上采样方式
    aspp_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    aspp_pool = tf.keras.layers.Reshape((1, 1, -1))(aspp_pool)
    aspp_pool = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False)(aspp_pool)  # 修改为128保持一致
    aspp_pool = tf.keras.layers.BatchNormalization()(aspp_pool)
    aspp_pool = tf.keras.layers.Activation('relu')(aspp_pool)
    
    # 根据x的实际尺寸动态上采样
    aspp_pool = tf.keras.layers.UpSampling2D(
        size=(x_shape[1], x_shape[2]),  # 直接使用x的高度和宽度
        interpolation='bilinear'
    )(aspp_pool)

    # 合并ASPP分支
    aspp_concat = tf.keras.layers.Concatenate()([aspp_conv1, aspp_conv2, aspp_conv3, aspp_pool])
    aspp_concat = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False)(aspp_concat)  # 从256减少到128通道
    
    # 上采样ASPP特征
    aspp_upsampled = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                interpolation='bilinear')(aspp_concat)

    # 计算上采样比例
    upsample_ratio = aspp_upsampled.shape[1] // low_level_features.shape[1]
    if upsample_ratio < 1:
        upsample_ratio = 1

    # 上采样低级特征以匹配ASPP特征的尺寸
    low_level_features = tf.keras.layers.UpSampling2D(
        size=(upsample_ratio, upsample_ratio),
        interpolation='bilinear'
    )(low_level_features)

    # 如果尺寸不匹配，使用padding或crop进行调整
    if low_level_features.shape[1] != aspp_upsampled.shape[1]:
        low_level_features = tf.keras.layers.Cropping2D(
            cropping=((0, low_level_features.shape[1] - aspp_upsampled.shape[1]),
                     (0, low_level_features.shape[2] - aspp_upsampled.shape[2]))
        )(low_level_features)

    # 合并低级特征和ASPP特征
    decoder_concat = tf.keras.layers.Concatenate()([aspp_upsampled, low_level_features])

    # 解码器，移除 dropout
    decoder = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(decoder_concat)  # 从256减少到128通道
    decoder = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(decoder)  # 从256减少到128通道

    # 计算最终上采样比例
    final_upsample_ratio = input_shape[0] // decoder.shape[1]
    
    # 最终上采样和预测
    decoder = tf.keras.layers.UpSampling2D(size=(final_upsample_ratio, final_upsample_ratio),
                                         interpolation='bilinear')(decoder)
    # 修改输出通道数为1
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same')(decoder)

    # 创建模型
    model = tf.keras.Model(inputs=input_image, outputs=outputs)
    return model

# 3. 定义损失函数和评估指标
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.keras.backend.sigmoid(y_pred), tf.float32)
    
    # 确保输入是4D张量 [batch_size, height, width, channels]
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    # 展平预测值和真实值
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(tf.keras.backend.sigmoid(y_pred), epsilon, 1. - epsilon)
    
    # 确保输入是4D张量
    y_true = tf.cast(y_true, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    
    # 计算focal loss
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    focal_loss = - alpha_t * tf.pow(1. - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)

def combined_loss(y_true, y_pred):
    # 确保输入是4D张量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    return dice_loss(y_true, y_pred) + binary_focal_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    # 确保输入是4D张量
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.keras.backend.sigmoid(y_pred) > 0.5, tf.float32)
    
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    # 展平预测值和真实值
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def train():
    # 添加内存清理
    gc.collect()
    
    # 限制TensorFlow内存增长
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            # 预留4GB给系统
            tf.config.experimental.set_memory_growth(device, True)
            # 限制显存使用上限
            try:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=26000)]  # 限制为26GB
                )
            except RuntimeError as e:
                # 捕获已经初始化的运行时错误
                print(e)
    
    # 启用XLA加速
    tf.config.optimizer.set_jit(True)
    
    # V100性能强劲，但为稳定性略微降低学习率
    initial_learning_rate = 3e-4  # 调低学习率以稳定训练
    
    # 使用标准优化器而非legacy版本
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # 直接创建模型，不使用strategy.scope()
    model = DeepLabV3Plus()
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coef, iou_metric]
    )

    # 更激进的回调函数设置
    callbacks = [
        ModelCheckpoint(
            'deeplabv3plus_mbv3_best.keras',
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=2,  # 更快降低学习率
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=6,  # 更早停止
            restore_best_weights=True,
            verbose=1
        )
    ]

    # CPU使用率优化
    import threading
    
    # 优化点5: 限制TensorFlow使用的线程数
    tf.config.threading.set_inter_op_parallelism_threads(4)  # 限制线程池大小
    tf.config.threading.set_intra_op_parallelism_threads(4)  # 限制操作内并行度
    
    # 优化点6: 设置内存增长策略
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # 显存动态增长，避免一次性分配过多
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,  # 指定训练步数
        validation_data=val_dataset,      # 验证集是有限的，不需要指定steps
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
    model.save('deeplabv3plus_mbv3_final.keras')
    print("模型训练完成并保存")
    
    return train_dataset  # 返回训练数据集供convert函数使用

def convert():
    print("开始转换TFLite模型...")
    # 加载最佳模型
    model = tf.keras.models.load_model(
        'deeplabv3plus_mbv3_best.keras',
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

    # 创建推断模型
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocessed = tf.keras.layers.Lambda(preprocess_input)(input_tensor)
    features = model(preprocessed)
    outputs = tf.keras.layers.Lambda(post_processing)(features)
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    # TFLite转换
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # 修改这里：添加额外的配置
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 添加对基本运算的支持
        tf.lite.OpsSet.SELECT_TF_OPS     # 添加对TF运算的支持
    ]
    
    # 量化配置
    def representative_dataset():
        for image_batch, _ in train_dataset.take(100):
            for image in image_batch:
                image = tf.expand_dims(image, axis=0)
                yield [image]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 转换并保存模型
    tflite_model = converter.convert()
    with open('doc_scanner_mbv3.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite模型已保存")
    print(f"TFLite模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")

def convert_test():
    """
    用于测试的TFLite转换函数，去掉量化步骤以加快转换速度
    """
    print("1. 开始加载模型...")
    model = tf.keras.models.load_model(
        'deeplabv3plus_mbv3_best.keras',
        custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'binary_focal_loss': binary_focal_loss,
            'combined_loss': combined_loss,
            'iou_metric': iou_metric
        }
    )

    print("2. 创建推断模型...")
    # 创建简单的推断模型
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    outputs = model(input_tensor)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)  # 添加sigmoid激活函数
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    print("3. 配置转换器...")
    # 简单的TFLite转换配置
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    print("4. 开始转换(这可能需要几分钟)...")
    tflite_model = converter.convert()

    print("5. 保存模型...")
    with open('doc_scanner_mbv3_test.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("转换完成!")
    print(f"TFLite模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    return 'doc_scanner_mbv3_test.tflite'

def test_tflite_model(tflite_model_path='doc_scanner_mbv3_test.tflite'):
    """测试TFLite模型的推理效果"""
    print("开始测试TFLite模型...")
    
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 获取输入输出细节
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n模型信息:")
    print("输入详情:", input_details)
    print("输出详情:", output_details)

    # 从验证集获取一个样本进行测试
    for images, masks in val_dataset.take(1):
        test_image = images[0]
        test_mask = masks[0]
        break

    # 预处理图像
    input_shape = input_details[0]['shape']
    test_image_uint8 = tf.cast(test_image * 255, tf.uint8).numpy()
    
    # 设置输入张量
    interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(test_image_uint8, 0))
    
    # 运行推理
    print("\n执行推理...")
    interpreter.invoke()
    
    # 获取输出
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.astype(np.float32) / 255.0  # 从uint8转回float32
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('original image')
    plt.axis('off')
    
    # 显示真实掩码
    plt.subplot(1, 3, 2)
    plt.imshow(test_mask, cmap='gray')
    plt.title('ground truth mask')
    plt.axis('off')
    
    # 显示预测结果
    plt.subplot(1, 3, 3)
    plt.imshow(output[0,:,:,0], cmap='gray')
    plt.title('TFLite prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('tflite_test_result.png')
    plt.show()
    
    print("\n测试完成! 结果已保存为 'tflite_test_result.png'")
    
    # 计算性能指标
    pred_mask = output[0,:,:,0] > 0.5
    true_mask = test_mask.numpy() > 0.5
    
    iou = np.sum(pred_mask & true_mask) / np.sum(pred_mask | true_mask)
    print(f"\n测试样本的IoU: {iou:.4f}")

# 使用示例:
if __name__ == "__main__":
    # 设置环境变量以减少警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 获取数据路径并创建数据集
    image_paths, mask_paths = get_dataset_paths()
    
    # 分割训练和验证集
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42)
    
    # 计算steps_per_epoch
    steps_per_epoch = len(train_img_paths) // BATCH_SIZE
    steps_per_epoch = max(1, steps_per_epoch)
    
    print(f"训练集大小: {len(train_img_paths)}")
    print(f"验证集大小: {len(val_img_paths)}")
    print(f"每轮训练步数: {steps_per_epoch}")
    
    # 创建数据集时指定sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    train_dataset = create_dataset(train_img_paths, train_mask_paths, training=True)
    train_dataset = train_dataset.with_options(options)
    
    val_dataset = create_dataset(val_img_paths, val_mask_paths, training=False)
    val_dataset = val_dataset.with_options(options)
    
    # 在数据加载部分添加一个简单的过滤器减少训练数据量进行测试
    if TRAIN_LEN == 0:  # 如果未指定训练长度
        # 可以先设置一个较小的子集进行测试
        TRAIN_SUBSET = 1000  # 先用1000张图像训练测试速度
        if len(train_img_paths) > TRAIN_SUBSET:
            train_img_paths = train_img_paths[:TRAIN_SUBSET]
            train_mask_paths = train_mask_paths[:TRAIN_SUBSET]
    
    # 训练模型
    train()
    
    # 转换为TFLite
    # convert()
    convert_test()
    
    # 测试TFLite模型
    test_tflite_model()