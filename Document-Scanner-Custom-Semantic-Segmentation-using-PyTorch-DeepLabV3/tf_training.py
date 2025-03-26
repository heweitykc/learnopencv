# -*- coding: utf-8 -*-
"""Custom_training_deeplabv3_mbv3_TFLite.ipynb"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 在导入TensorFlow前设置关键环境变量 - 优化V100单卡训练
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少日志输出
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # 动态显存分配
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'  # 强制XLA优化
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # GPU线程模式优化
os.environ['TF_CUDNN_DETERMINISTIC'] = '0'  # 禁用确定性，提高性能

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import gc

# V100 32GB显存优化配置
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # 使用混合精度训练

# 早期TF线程配置 - 必须在任何TF操作前设置
tf.config.threading.set_inter_op_parallelism_threads(4)  # 12核CPU优化
tf.config.threading.set_intra_op_parallelism_threads(6)  # 预留部分核心给系统

# 模型训练参数 - 针对V100单卡 + 12核CPU优化
TRAIN_LEN = 0
VAL_LEN = 0
EPOCHS = 3
NUM_CLASSES = 2

# 更平衡的配置 - 针对384*480原始图像
IMG_SIZE = 416       # 略微提高但不过度上采样
BATCH_SIZE = 64      # 大幅提高批量大小充分利用V100显存

# GPU配置优化
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到GPU: {len(gpus)}个")
    except RuntimeError as e:
        print(e)

train_dataset = None
val_dataset = None

# 数据准备函数保持不变
def get_dataset_paths(images_dir='/mnt/data/scan/document_dataset_resized/train/images', 
                      masks_dir='/mnt/data/scan/document_dataset_resized/train/masks'):
    image_paths = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))

    if TRAIN_LEN > 0 and len(image_paths) > TRAIN_LEN + VAL_LEN:
        image_paths = image_paths[:TRAIN_LEN + VAL_LEN]
        mask_paths = mask_paths[:TRAIN_LEN + VAL_LEN]

    print(f"找到图像: {len(image_paths)}张")
    print(f"找到掩码: {len(mask_paths)}张")
    assert len(image_paths) == len(mask_paths), "图像和掩码数量不匹配!"

    return image_paths, mask_paths

# 针对12核CPU优化的数据增强 - 保持基本增强但减少计算量
@tf.function
def optimized_data_augmentation(image, mask):
    # 随机翻转 - 计算简单的数据增强
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
        mask = tf.image.random_flip_left_right(mask)
    
    # 简化的亮度调整 - 仅使用一种增强减轻CPU负担
    image = tf.image.random_brightness(image, 0.1)    
    image = tf.clip_by_value(image, 0, 1)
    
    return image, mask

# 图像解析函数 - 优化resize操作
@tf.function
def parse_image(img_path, mask_path):
    # 读取图像 - 使用nearest邻近插值加速处理
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], 
                          method='bilinear')  # 高质量调整
    image = tf.cast(image, tf.float32) / 255.0

    # 读取掩码 - 使用快速的nearest插值
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], 
                         method='nearest')  # 掩码用nearest更合适
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask

# V100+12核CPU优化的数据加载函数
def create_dataset(image_paths, mask_paths, training=True):
    AUTOTUNE = tf.data.AUTOTUNE
    
    # 最佳配置选项
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic = False  # 提高性能
    options.experimental_optimization.map_parallelization = True
    
    # 重要: 禁用可能增加CPU负担的高级优化
    options.experimental_optimization.map_and_batch_fusion = False
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.with_options(options)
    
    # 针对12核CPU优化的并行处理设置
    if training:
        # 适中的shuffle buffer - 根据92GB内存调整
        dataset = dataset.shuffle(buffer_size=2000)
        
        # 减少并行调用数 - 避免CPU过载
        dataset = dataset.map(parse_image, num_parallel_calls=4)
        dataset = dataset.map(optimized_data_augmentation, num_parallel_calls=4)
        
        # 使用适中的批量大小
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.repeat()  # 放在batch后面可减少内存使用
    else:
        dataset = dataset.map(parse_image, num_parallel_calls=2)  # 验证集减少并行度
        dataset = dataset.batch(BATCH_SIZE)
    
    # 预取操作 - 减小预取量降低内存压力
    dataset = dataset.prefetch(buffer_size=2)  # 固定小预取值减轻内存压力
    
    return dataset

# DeepLabV3+ 模型 - 针对32GB V100显存优化
def DeepLabV3Plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # 基础模型 - MobileNetV3Large
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        minimalistic=True
    )
    
    # 打印层名称以便调试
    print("模型层名称:")
    for i, layer in enumerate(base_model.layers):
        if "project" in layer.name and i < 50:  # 只打印前面的层
            print(f"{i}: {layer.name}")
    
    # 加载预训练权重
    temp_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        minimalistic=True
    )
    
    # 复制权重
    for layer, temp_layer in zip(base_model.layers, temp_model.layers):
        if layer.name == temp_layer.name:
            try:
                layer.set_weights(temp_layer.get_weights())
            except ValueError:
                print(f"跳过层 {layer.name} 的权重加载")
    
    # 针对V100性能冻结较少的早期层 - 提高模型表达能力
    for layer in base_model.layers[:80]:  # 从100减少到80
        layer.trainable = False

    # 获取特征图
    input_image = base_model.input
    
    # 修改这里：使用正确的层名称 'expanded_conv_3/project' 替代 'expanded_conv_3_project'
    low_level_features = base_model.get_layer('expanded_conv_3/project').output
    
    # 增加特征通道 - 利用32GB显存提高表达能力
    low_level_features = tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False)(low_level_features)  # 从64增加到128
    low_level_features = tf.keras.layers.BatchNormalization()(low_level_features)
    low_level_features = tf.keras.layers.Activation('relu')(low_level_features)

    # ASPP - 利用V100增强处理能力
    x = base_model.output
    x_shape = tf.keras.backend.int_shape(x)
    
    # 增加特征通道数 - 利用32GB显存提高表达能力
    aspp_conv1 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)  # 从160增加到256
    aspp_conv1 = tf.keras.layers.BatchNormalization()(aspp_conv1)
    aspp_conv1 = tf.keras.layers.Activation('relu')(aspp_conv1)

    # 空洞卷积
    aspp_conv2 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    aspp_conv2 = tf.keras.layers.BatchNormalization()(aspp_conv2)
    aspp_conv2 = tf.keras.layers.Activation('relu')(aspp_conv2)

    aspp_conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    aspp_conv3 = tf.keras.layers.BatchNormalization()(aspp_conv3)
    aspp_conv3 = tf.keras.layers.Activation('relu')(aspp_conv3)

    # 增加额外空洞卷积分支
    aspp_conv4 = tf.keras.layers.Conv2D(256, 3, padding='same', dilation_rate=18, use_bias=False)(x)  # 新增分支

    # 全局池化
    aspp_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    aspp_pool = tf.keras.layers.Reshape((1, 1, -1))(aspp_pool)
    aspp_pool = tf.keras.layers.Conv2D(160, 1, padding='same', use_bias=False)(aspp_pool)
    aspp_pool = tf.keras.layers.BatchNormalization()(aspp_pool)
    aspp_pool = tf.keras.layers.Activation('relu')(aspp_pool)
    
    aspp_pool = tf.keras.layers.UpSampling2D(
        size=(x_shape[1], x_shape[2]),
        interpolation='bilinear'
    )(aspp_pool)

    # 合并ASPP分支
    aspp_concat = tf.keras.layers.Concatenate()([aspp_conv1, aspp_conv2, aspp_conv3, aspp_conv4, aspp_pool])  # 添加新分支
    aspp_concat = tf.keras.layers.Conv2D(160, 1, padding='same', use_bias=False)(aspp_concat)
    
    # 上采样ASPP特征
    aspp_upsampled = tf.keras.layers.UpSampling2D(size=(4, 4),
                                                interpolation='bilinear')(aspp_concat)

    # 确保特征尺寸匹配
    upsample_ratio = aspp_upsampled.shape[1] // low_level_features.shape[1]
    if upsample_ratio < 1:
        upsample_ratio = 1

    low_level_features = tf.keras.layers.UpSampling2D(
        size=(upsample_ratio, upsample_ratio),
        interpolation='bilinear'
    )(low_level_features)

    if low_level_features.shape[1] != aspp_upsampled.shape[1]:
        low_level_features = tf.keras.layers.Cropping2D(
            cropping=((0, low_level_features.shape[1] - aspp_upsampled.shape[1]),
                     (0, low_level_features.shape[2] - aspp_upsampled.shape[2]))
        )(low_level_features)

    # 合并特征
    decoder_concat = tf.keras.layers.Concatenate()([aspp_upsampled, low_level_features])

    # 解码器 - 利用V100增加特征通道
    decoder = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(decoder_concat)
    decoder = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(decoder)
    decoder = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(decoder)  # 额外增加一层

    # 最终上采样
    final_upsample_ratio = input_shape[0] // decoder.shape[1]
    decoder = tf.keras.layers.UpSampling2D(size=(final_upsample_ratio, final_upsample_ratio),
                                         interpolation='bilinear')(decoder)
    
    # 输出层
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same')(decoder)

    # 创建模型
    model = tf.keras.Model(inputs=input_image, outputs=outputs)
    return model

# 损失函数和评估指标保持不变
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.keras.backend.sigmoid(y_pred), tf.float32)
    
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(tf.keras.backend.sigmoid(y_pred), epsilon, 1. - epsilon)
    
    y_true = tf.cast(y_true, tf.float32)
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    
    focal_loss = - alpha_t * tf.pow(1. - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(focal_loss)

def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    return dice_loss(y_true, y_pred) + binary_focal_loss(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.keras.backend.sigmoid(y_pred) > 0.5, tf.float32)
    
    if len(y_true.shape) == 3:
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=-1)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 针对单V100优化的训练函数
def train():
    # 内存清理
    gc.collect()
    
    # 启用XLA加速 - 优化V100性能
    tf.config.optimizer.set_jit(True)
    
    # 检查GPU配置 - 适应单V100
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"训练将使用 {len(physical_devices)} 个GPU")
        for i, device in enumerate(physical_devices):
            print(f"  GPU #{i}: {device.name}")
    else:
        print("警告: 未检测到GPU，训练将使用CPU")
    
    # 按比例调整学习率，匹配更大的批量大小
    initial_learning_rate = 8e-4  # 从4e-4调整到8e-4，因为批量大小也提高了
    
    # 模型创建
    model = DeepLabV3Plus()
    
    # 优化器配置
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coef, iou_metric]
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            'deeplabv3plus_mbv3_best_weights.h5',  # 只保存权重而不是整个模型
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_weights_only=True  # 关键修改：只保存权重
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=3,
            min_lr=5e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq=100,
            profile_batch=0
        )
    ]
    
    # 更高效的数据集创建
    def create_simple_training_dataset():
        global train_img_paths, train_mask_paths
        
        dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
        dataset = dataset.shuffle(buffer_size=5000)  # 增大shuffle缓冲区
        dataset = dataset.map(parse_image, num_parallel_calls=8)  # 增加并行处理
        dataset = dataset.batch(BATCH_SIZE)  # 大批量处理
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)  # 让TF自动决定预取量
        return dataset
    
    def create_simple_validation_dataset():
        global val_img_paths, val_mask_paths
        
        # 创建简单验证数据集
        dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))
        dataset = dataset.map(parse_image, num_parallel_calls=2)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=1)
        return dataset
    
    # 创建没有options的简单数据集
    simple_train_dataset = create_simple_training_dataset()
    simple_val_dataset = create_simple_validation_dataset()
    
    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        simple_train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=simple_val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # 只保存权重，避免复杂模型保存问题
    model.save_weights('deeplabv3plus_mbv3_final_weights.h5')
    print("模型权重已保存")
    
    # 保存最佳权重的模型用于TFLite转换
    model.load_weights('deeplabv3plus_mbv3_best_weights.h5')
    print("已加载最佳模型权重")
    
    return model  # 返回加载了最佳权重的模型

# 优化的convert函数 - 大幅提高速度，适合开发环境
def convert(model=None):
    print("开始转换TFLite模型(动态范围量化)...")
    
    # 如果没有传入模型，尝试重新创建一个并加载权重
    if model is None:
        print("未传入模型，尝试创建新模型并加载权重...")
        model = DeepLabV3Plus()
        try:
            model.load_weights('deeplabv3plus_mbv3_best_weights.h5')
            print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return None

    # 创建推断模型
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    outputs = model(input_tensor)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    # 使用更快的动态范围量化 - 不需要representative_dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 仍保留优化但使用动态范围量化
    
    # 支持的操作和平台
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    print("正在转换模型(使用动态范围量化)...")
    tflite_model = converter.convert()
    
    # 保存模型
    with open('doc_scanner_mbv3.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite模型已保存")
    print(f"TFLite模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    # 保存到指定目录
    import os
    import datetime
    
    save_dir = "/mnt/data/scan/"
    os.makedirs(save_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tflite_filename = f"doc_scanner_mbv3_{current_time}.tflite"
    tflite_filepath = os.path.join(save_dir, tflite_filename)
    
    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite模型已保存到: {tflite_filepath}")

    return 'doc_scanner_mbv3.tflite'

# 保留完整量化的函数 - 仅用于最终产品发布
def convert_production(model=None):
    print("开始生成生产版TFLite模型(完整INT8量化)...")
    
    if model is None:
        print("未传入模型，尝试创建新模型并加载权重...")
        model = DeepLabV3Plus()
        try:
            model.load_weights('deeplabv3plus_mbv3_best_weights.h5')
            print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return None

    # 创建推断模型
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    outputs = model(input_tensor)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    # 完整量化配置
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 只使用10张图像进行量化 - 减少时间但仍保持质量
    def create_minimal_dataset_for_quantization():
        global train_img_paths
        paths = train_img_paths[:10]  # 从100减少到10张
        
        def preprocess_image(img_path):
            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            image = tf.cast(image, tf.float32) / 255.0
            return image
            
        dataset = tf.data.Dataset.from_tensor_slices(paths)
        dataset = dataset.map(preprocess_image)
        dataset = dataset.batch(1)
        return dataset
    
    # 使用最小数据集进行量化
    simple_dataset = create_minimal_dataset_for_quantization()
    def representative_dataset():
        for image in simple_dataset:
            yield [image]

    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 执行转换
    print("正在转换生产版模型(可能需要几分钟)...")
    tflite_model = converter.convert()
    
    # 保存模型
    production_path = 'doc_scanner_mbv3_production.tflite'
    with open(production_path, 'wb') as f:
        f.write(tflite_model)
    print(f"生产版TFLite模型已保存: {production_path}")
    print(f"生产版模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    # 保存到指定目录
    import os
    import datetime
    
    save_dir = "/mnt/data/scan/"
    os.makedirs(save_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tflite_filename = f"doc_scanner_mbv3_production_{current_time}.tflite"
    tflite_filepath = os.path.join(save_dir, tflite_filename)
    
    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"生产版TFLite模型已保存到: {tflite_filepath}")

    return production_path

# 非量化TFLite转换函数 - 用于快速测试
def convert_test(model=None):
    print("开始快速转换TFLite模型(无量化)...")
    
    # 如果没有传入模型，尝试重新创建一个并加载权重
    if model is None:
        print("未传入模型，尝试创建新模型并加载权重...")
        model = DeepLabV3Plus()
        try:
            model.load_weights('deeplabv3plus_mbv3_best_weights.h5')
            print("成功加载模型权重")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return None

    # 创建推断模型
    input_tensor = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    outputs = model(input_tensor)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    inference_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    # 最简TFLite转换配置 - 不使用量化以加快处理
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # 只支持基本操作，但跳过复杂量化
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 执行转换 - 应该比量化版本快很多
    print("正在快速转换模型...")
    tflite_model = converter.convert()
    
    # 保存不同名称避免混淆
    test_model_path = 'doc_scanner_mbv3_test.tflite'
    with open(test_model_path, 'wb') as f:
        f.write(tflite_model)
    print("测试TFLite模型已保存")
    print(f"测试TFLite模型大小: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    # 也保存一份到时间戳目录，但带test标记
    import os
    import datetime
    
    save_dir = "/mnt/data/scan/"
    os.makedirs(save_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tflite_filename = f"doc_scanner_mbv3_test_{current_time}.tflite"
    tflite_filepath = os.path.join(save_dir, tflite_filename)
    
    with open(tflite_filepath, 'wb') as f:
        f.write(tflite_model)
    
    print(f"测试TFLite模型已保存到: {tflite_filepath}")

    return test_model_path  # 返回模型路径以便进一步测试

# 主函数优化
if __name__ == "__main__":
    print("\n=== 开始文档扫描分割模型训练 ===")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"GPU配置: V100 (32GB)")
    print(f"混合精度训练: {'已启用' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else '未启用'}")
    print(f"训练分辨率: {IMG_SIZE}x{IMG_SIZE}")
    print(f"批量大小: {BATCH_SIZE}")
    
    # 获取数据路径
    image_paths, mask_paths = get_dataset_paths()
    
    # 训练验证分割 - 增加训练集比例
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42)
    
    # 根据实际数据集大小计算steps_per_epoch
    steps_per_epoch = len(train_img_paths) // BATCH_SIZE
    steps_per_epoch = max(1, steps_per_epoch)
    
    print(f"训练集大小: {len(train_img_paths)}张图像")
    print(f"验证集大小: {len(val_img_paths)}张图像")
    print(f"每轮训练步数: {steps_per_epoch}")
    
    # 创建优化的数据集
    print("\n准备数据集中...")
    train_dataset = create_dataset(train_img_paths, train_mask_paths, training=True)
    val_dataset = create_dataset(val_img_paths, val_mask_paths, training=False)
    
    # 训练模型并获取训练好的模型
    print("\n配置完成，开始训练...")
    trained_model = train()
    
    # 转换为测试版模型 - 无量化，超快
    # print("\n转换为测试TFLite模型(无量化)...")
    # test_model_path = convert_test(trained_model)
    
    # 转换为开发版模型 - 使用动态范围量化，速度适中
    print("\n转换为开发版TFLite模型(动态范围量化)...")
    dev_model_path = convert(trained_model)
    
    # 如果需要生产版模型，可以取消注释下面的代码
    # print("\n转换为生产版TFLite模型(完整INT8量化)...")
    # prod_model_path = convert_production(trained_model)
    
    print("\n=== 训练和转换过程完成 ===")