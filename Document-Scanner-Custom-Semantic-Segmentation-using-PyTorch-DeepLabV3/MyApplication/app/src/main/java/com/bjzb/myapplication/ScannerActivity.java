package com.bjzb.myapplication;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.bjzb.myapplication.utils.DocumentDetector;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class ScannerActivity extends AppCompatActivity {
    private static final String TAG = "ScannerActivity";

    // 权限请求码
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    // 屏幕旋转方向
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    // 相机预览
    private TextureView previewView;
    private View documentOverlay; // 用于绘制文档边缘的覆盖层
    private TextView tvGuide;
    private View scanningIndicator;

    // 控制按钮
    private ImageButton btnCapture;
    private ImageButton btnGallery;
    private ImageButton btnFlash;

    // 相机相关变量
    private String cameraId;
    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;
    private CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private ImageReader imageReader;

    // 处理线程
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private Semaphore cameraOpenCloseLock = new Semaphore(1);

    // 闪光灯状态
    private boolean isFlashSupported = false;
    private boolean isFlashOn = false;

    // 文档边缘检测相关
    private boolean isDocumentDetectionEnabled = true; // 是否启用文档检测
    private long lastProcessingTimeMs = 0;
    private static final long MIN_PROCESSING_INTERVAL_MS = 500; // 处理间隔
    private Point[] documentCorners = null; // 检测到的文档角点

    // OpenCV加载状态
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV加载失败");
        } else {
            Log.d(TAG, "OpenCV加载成功");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_scanner);

        // 初始化视图
        previewView = findViewById(R.id.previewView);
        documentOverlay = findViewById(R.id.documentOverlay);
        tvGuide = findViewById(R.id.tvGuide);
        scanningIndicator = findViewById(R.id.scanningIndicator);

        btnCapture = findViewById(R.id.btnCapture);
        btnGallery = findViewById(R.id.btnGallery);
        btnFlash = findViewById(R.id.btnFlash);

        // 设置TextureView的表面纹理监听器
        previewView.setSurfaceTextureListener(textureListener);

        // 设置按钮点击事件
        btnCapture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                takePicture();
            }
        });

        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        btnFlash.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                toggleFlash();
            }
        });

        // 初始状态
        scanningIndicator.setVisibility(View.GONE);
    }

    /**
     * TextureView的表面纹理监听器
     */
    private TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
            // 当Surface可用时，打开相机
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {
            // 处理Surface尺寸变化，如屏幕旋转
        }

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
            // 预览每一帧更新时的回调
            if (isDocumentDetectionEnabled) {
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastProcessingTimeMs > MIN_PROCESSING_INTERVAL_MS) {
                    lastProcessingTimeMs = currentTime;
                    detectDocumentInPreview();
                }
            }
        }
    };

    /**
     * 相机状态回调
     */
    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            // 相机打开成功
            Log.d(TAG, "相机打开成功");
            cameraOpenCloseLock.release();
            cameraDevice = camera;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            // 相机断开连接
            cameraOpenCloseLock.release();
            camera.close();
            cameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            // 相机出错
            cameraOpenCloseLock.release();
            camera.close();
            cameraDevice = null;
            finish();
        }
    };

    /**
     * 打开相机
     */
    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            cameraId = getBackFacingCameraId(manager);
            if (cameraId == null) {
                Toast.makeText(this, "无法找到后置相机", Toast.LENGTH_SHORT).show();
                finish();
                return;
            }

            // 获取相机特性
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

            // 获取相机支持的尺寸
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map == null) {
                throw new IllegalStateException("无法获取相机配置");
            }

            // 选择合适的预览尺寸
            imageDimension = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                    previewView.getWidth(), previewView.getHeight());

            // 检查是否支持闪光灯
            isFlashSupported = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) != null &&
                    characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);

            // 更新闪光灯按钮状态
            updateFlashButtonVisibility();

            // 检查权限
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
                return;
            }

            // 创建图像读取器
            setupImageReader();

            // 确保可以获取相机访问权
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("获取相机访问权超时");
            }

            // 打开相机
            manager.openCamera(cameraId, stateCallback, backgroundHandler);

        } catch (CameraAccessException e) {
            Log.e(TAG, "相机访问异常: " + e.getMessage());
        } catch (InterruptedException e) {
            Log.e(TAG, "打开相机时中断: " + e.getMessage());
        } catch (IllegalStateException e) {
            Log.e(TAG, "相机状态错误: " + e.getMessage());
        }
    }

    /**
     * 获取后置相机ID
     */
    private String getBackFacingCameraId(CameraManager manager) throws CameraAccessException {
        for (String cameraId : manager.getCameraIdList()) {
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
            if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                return cameraId;
            }
        }
        return null;  // 没有找到后置相机
    }

    /**
     * 设置图像读取器
     */
    private void setupImageReader() {
        // 创建适合拍照的图像读取器
        imageReader = ImageReader.newInstance(1920, 1080, ImageFormat.JPEG, 1);
        imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Image image = null;
                try {
                    image = reader.acquireLatestImage();
                    ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                    byte[] bytes = new byte[buffer.capacity()];
                    buffer.get(bytes);
                    processImage(bytes);
                } finally {
                    if (image != null) {
                        image.close();
                    }
                }
            }
        }, backgroundHandler);
    }

    /**
     * 选择最优的预览尺寸
     */
    private Size chooseOptimalSize(Size[] choices, int width, int height) {
        // 按面积排序
        List<Size> bigEnough = new ArrayList<>();
        for (Size option : choices) {
            if (option.getHeight() == option.getWidth() * height / width &&
                    option.getWidth() >= width && option.getHeight() >= height) {
                bigEnough.add(option);
            }
        }

        // 如果找到了符合条件的尺寸，选择最小的一个
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, (o1, o2) -> Long.signum(
                    (long) o1.getWidth() * o1.getHeight() - (long) o2.getWidth() * o2.getHeight()));
        }

        // 如果没有找到符合比例的尺寸，选择最接近的一个
        return choices[0];
    }

    /**
     * 创建相机预览
     */
    private void createCameraPreview() {
        try {
            SurfaceTexture texture = previewView.getSurfaceTexture();
            if (texture == null) {
                return;
            }

            // 设置预览尺寸
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);

            // 创建预览请求
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            // 创建相机预览会话
            cameraDevice.createCaptureSession(Arrays.asList(surface, imageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            if (cameraDevice == null) {
                                return;
                            }

                            // 当会话配置完成，开始预览
                            cameraCaptureSession = session;
                            updatePreview();
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Toast.makeText(ScannerActivity.this, "相机配置失败", Toast.LENGTH_SHORT).show();
                        }
                    }, null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "创建预览时出错: " + e.getMessage());
        }
    }

    /**
     * 更新相机预览
     */
    private void updatePreview() {
        if (cameraDevice == null) {
            return;
        }

        try {
            // 设置自动对焦模式
            captureRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            // 设置闪光灯状态
            setFlashMode(captureRequestBuilder);

            // 开始预览
            cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler);
        } catch (CameraAccessException e) {
            Log.e(TAG, "更新预览时出错: " + e.getMessage());
        }
    }

    /**
     * 设置闪光灯模式
     */
    private void setFlashMode(CaptureRequest.Builder requestBuilder) {
        if (isFlashSupported) {
            if (isFlashOn) {
                requestBuilder.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH);
            } else {
                requestBuilder.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF);
            }
        }
    }

    /**
     * 更新闪光灯按钮可见性
     */
    private void updateFlashButtonVisibility() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (isFlashSupported) {
                    btnFlash.setVisibility(View.VISIBLE);
                    btnFlash.setImageResource(isFlashOn ?
                            android.R.drawable.ic_menu_compass : android.R.drawable.ic_menu_compass);
                } else {
                    btnFlash.setVisibility(View.GONE);
                }
            }
        });
    }

    /**
     * 切换闪光灯状态
     */
    private void toggleFlash() {
        if (isFlashSupported) {
            isFlashOn = !isFlashOn;
            updateFlashButtonVisibility();
            updatePreview();
        }
    }

    /**
     * 拍照
     */
    private void takePicture() {
        if (cameraDevice == null) {
            return;
        }

        try {
            // 显示扫描指示器
            scanningIndicator.setVisibility(View.VISIBLE);

            // 创建拍照请求
            final CaptureRequest.Builder captureBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(imageReader.getSurface());

            // 设置自动对焦模式
            captureBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

            // 设置闪光灯模式
            setFlashMode(captureBuilder);

            // 设置旋转方向
            int rotation = getWindowManager().getDefaultDisplay().getRotation();
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, ORIENTATIONS.get(rotation));

            // 捕获图像
            cameraCaptureSession.stopRepeating();
            cameraCaptureSession.capture(captureBuilder.build(), new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                               @NonNull CaptureRequest request,
                                               @NonNull android.hardware.camera2.TotalCaptureResult result) {
                    super.onCaptureCompleted(session, request, result);
                    try {
                        // 恢复预览
                        cameraCaptureSession.setRepeatingRequest(
                                captureRequestBuilder.build(), null, backgroundHandler);
                    } catch (CameraAccessException e) {
                        Log.e(TAG, "恢复预览失败: " + e.getMessage());
                    }
                }
            }, null);

        } catch (CameraAccessException e) {
            Log.e(TAG, "拍照时出错: " + e.getMessage());
        }
    }

    /**
     * 打开相册
     */
    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, 100);
    }

    /**
     * 处理拍摄的图像
     */
    private void processImage(byte[] imageData) {
        // 将图像数据转为位图
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);

        // 隐藏扫描指示器
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                scanningIndicator.setVisibility(View.GONE);
            }
        });

        // 启动结果处理Activity
        Intent intent = new Intent(this, ResultActivity.class);
        intent.putExtra("image_data", imageData);
        intent.putExtra("source", "camera");
        startActivity(intent);
    }

    /**
     * 在预览中检测文档
     */
    private void detectDocumentInPreview() {
        if (previewView == null || !previewView.isAvailable()) {
            return;
        }

        backgroundHandler.post(new Runnable() {
            @Override
            public void run() {
                try {
                    // 获取预览帧
                    Bitmap previewBitmap = previewView.getBitmap();
                    if (previewBitmap == null) {
                        return;
                    }

                    // 检测文档边缘
                    final Point[] corners = DocumentDetector.detectDocumentEdges(previewBitmap);

                    // 更新文档边缘绘制
                    documentCorners = corners;
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateDocumentOverlay();
                            updateGuideText(corners != null);
                        }
                    });

                } catch (Exception e) {
                    Log.e(TAG, "文档检测出错: " + e.getMessage());
                }
            }
        });
    }

    /**
     * 更新文档覆盖层
     */
    private void updateDocumentOverlay() {
        // 这里需要实现一个自定义View，绘制文档边缘
        // 为简化代码，这里省略实现
    }

    /**
     * 更新引导文本
     */
    private void updateGuideText(boolean documentDetected) {
        if (documentDetected) {
            tvGuide.setText("文档已识别，请保持稳定");
        } else {
            tvGuide.setText("将文档放入框内并保持稳定");
        }
    }

    /**
     * 开启后台线程
     */
    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("Camera Background");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    /**
     * 停止后台线程
     */
    private void stopBackgroundThread() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "停止后台线程时中断: " + e.getMessage());
            }
        }
    }

    /**
     * 关闭相机
     */
    private void closeCamera() {
        try {
            cameraOpenCloseLock.acquire();
            if (cameraCaptureSession != null) {
                cameraCaptureSession.close();
                cameraCaptureSession = null;
            }
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
            if (imageReader != null) {
                imageReader.close();
                imageReader = null;
            }
        } catch (InterruptedException e) {
            Log.e(TAG, "关闭相机时中断: " + e.getMessage());
        } finally {
            cameraOpenCloseLock.release();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (previewView.isAvailable()) {
                    openCamera();
                } else {
                    previewView.setSurfaceTextureListener(textureListener);
                }
            } else {
                Toast.makeText(this, "需要相机权限才能扫描文档", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 100 && resultCode == RESULT_OK && data != null) {
            // 从相册选择图片后启动结果Activity
            Intent intent = new Intent(this, ResultActivity.class);
            intent.setData(data.getData());
            intent.putExtra("source", "gallery");
            startActivity(intent);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (previewView.isAvailable()) {
            openCamera();
        } else {
            previewView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }
}