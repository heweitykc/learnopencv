package com.bjzb.myapplication.utils;

import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import android.media.ExifInterface;

/**
 * 图像处理工具类
 * 提供各种图像操作的实用方法
 */
public class ImageUtils {
    private static final String TAG = "ImageUtils";
    // 默认JPEG压缩质量
    private static final int DEFAULT_JPEG_QUALITY = 90;
    // 最大图像尺寸（防止OOM）
    private static final int MAX_IMAGE_DIMENSION = 2048;

    /**
     * 从Uri加载Bitmap图像
     */
    public static Bitmap getBitmapFromUri(Context context, Uri imageUri) throws IOException {
        // 获取输入流
        InputStream input = context.getContentResolver().openInputStream(imageUri);
        if (input == null) {
            throw new FileNotFoundException("无法打开文件流: " + imageUri);
        }

        // 先获取图像尺寸
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(input, null, options);
        input.close();

        // 计算采样率以避免OOM
        options.inSampleSize = calculateInSampleSize(options, MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION);
        options.inJustDecodeBounds = false;

        // 重新打开流并解码图像
        input = context.getContentResolver().openInputStream(imageUri);
        Bitmap bitmap = BitmapFactory.decodeStream(input, null, options);
        if (input != null) {
            input.close();
        }

        // 不再尝试从文件路径读取EXIF信息
        // 而是直接使用ExifInterface处理InputStream
        try (InputStream exifStream = context.getContentResolver().openInputStream(imageUri)) {
            if (exifStream != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                ExifInterface exif = new ExifInterface(exifStream);
                int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 
                                                     ExifInterface.ORIENTATION_NORMAL);
                bitmap = fixOrientation(bitmap, orientation);
            }
        } catch (Exception e) {
            Log.e(TAG, "读取EXIF信息失败: " + e.getMessage());
        }

        return bitmap;
    }

    /**
     * 从字节数组加载Bitmap图像
     */
    public static Bitmap getBitmapFromByteArray(byte[] data) {
        return getBitmapFromByteArray(data, MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION);
    }

    /**
     * 从字节数组加载Bitmap图像，具有最大尺寸限制
     */
    public static Bitmap getBitmapFromByteArray(byte[] data, int maxWidth, int maxHeight) {
        // 先获取图像尺寸
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeByteArray(data, 0, data.length, options);

        // 计算采样率
        options.inSampleSize = calculateInSampleSize(options, maxWidth, maxHeight);
        options.inJustDecodeBounds = false;

        // 解码图像
        return BitmapFactory.decodeByteArray(data, 0, data.length, options);
    }

    /**
     * 计算合适的图像采样率
     */
    private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            // 计算最大的inSampleSize值，该值是2的幂，并且可以使最终图像大于或等于请求的尺寸
            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }

        return inSampleSize;
    }

    /**
     * 从Uri获取真实文件路径
     */
    private static String getPathFromUri(Context context, Uri uri) {
        String result = null;
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = null;
        try {
            cursor = context.getContentResolver().query(uri, projection, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                result = cursor.getString(columnIndex);
            }
        } catch (Exception e) {
            Log.e(TAG, "获取文件路径出错: " + e.getMessage());
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        return result;
    }

    /**
     * 根据EXIF信息修正图像方向
     */
    private static Bitmap fixOrientation(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
            default:
                return bitmap;
        }
        
        try {
            Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, 
                                   bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            if (rotatedBitmap != bitmap) {
                bitmap.recycle();
                return rotatedBitmap;
            }
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "旋转图像内存不足: " + e.getMessage());
        }
        
        return bitmap;
    }

    /**
     * 压缩Bitmap为JPEG字节数组
     */
    public static byte[] compressBitmapToJpeg(Bitmap bitmap, int quality) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);
        return outputStream.toByteArray();
    }

    /**
     * 压缩Bitmap为JPEG字节数组（使用默认质量）
     */
    public static byte[] compressBitmapToJpeg(Bitmap bitmap) {
        return compressBitmapToJpeg(bitmap, DEFAULT_JPEG_QUALITY);
    }

    /**
     * 调整Bitmap大小
     */
    public static Bitmap resizeBitmap(Bitmap bitmap, int maxWidth, int maxHeight) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 计算缩放比例
        float scaleWidth = ((float) maxWidth) / width;
        float scaleHeight = ((float) maxHeight) / height;

        // 使用较小的缩放比例
        float scale = Math.min(scaleWidth, scaleHeight);

        // 如果不需要缩放
        if (scale >= 1.0f) {
            return bitmap;
        }

        // 创建变换矩阵
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);

        // 创建新的Bitmap
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }

    /**
     * 将BGR格式的Mat转换为RGB格式的Bitmap
     */
    public static Bitmap matToBitmap(Mat mat) {
        // 确保Mat类型正确
        Mat convertedMat = new Mat();
        if (mat.type() == CvType.CV_8UC1) {
            // 灰度图像转RGB
            Imgproc.cvtColor(mat, convertedMat, Imgproc.COLOR_GRAY2RGB);
        } else if (mat.type() == CvType.CV_8UC3) {
            // BGR转RGB
            Imgproc.cvtColor(mat, convertedMat, Imgproc.COLOR_BGR2RGB);
        } else if (mat.type() == CvType.CV_8UC4) {
            // BGRA转RGBA
            Imgproc.cvtColor(mat, convertedMat, Imgproc.COLOR_BGRA2RGBA);
        } else {
            // 其他类型，直接复制
            mat.copyTo(convertedMat);
        }

        // 创建位图
        Bitmap bitmap = Bitmap.createBitmap(convertedMat.cols(), convertedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(convertedMat, bitmap);
        convertedMat.release();

        return bitmap;
    }

    /**
     * 将RGB格式的Bitmap转换为BGR格式的Mat
     */
    public static Mat bitmapToMat(Bitmap bitmap) {
        Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
        Utils.bitmapToMat(bitmap, mat);

        // RGB到BGR转换
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2BGR);
        return mat;
    }

    /**
     * 将图像转换为灰度图
     */
    public static Bitmap convertToGrayscale(Bitmap originalBitmap) {
        // 创建新的空白位图
        Bitmap grayscaleBitmap = Bitmap.createBitmap(
                originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);

        // 创建Canvas和Paint
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();

        // 创建灰度ColorMatrix
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0); // 设置饱和度为0使其变为灰度

        // 应用ColorMatrix
        paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));

        // 绘制图像
        canvas.drawBitmap(originalBitmap, 0, 0, paint);

        return grayscaleBitmap;
    }

    /**
     * 提高图像对比度
     */
    public static Bitmap enhanceContrast(Bitmap originalBitmap, float contrast) {
        // 创建新的空白位图
        Bitmap enhancedBitmap = Bitmap.createBitmap(
                originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);

        // 创建Canvas和Paint
        Canvas canvas = new Canvas(enhancedBitmap);
        Paint paint = new Paint();

        // 创建对比度ColorMatrix
        ColorMatrix cm = new ColorMatrix(new float[]{
                contrast, 0, 0, 0, 0,
                0, contrast, 0, 0, 0,
                0, 0, contrast, 0, 0,
                0, 0, 0, 1, 0
        });

        // 应用ColorMatrix
        paint.setColorFilter(new ColorMatrixColorFilter(cm));

        // 绘制图像
        canvas.drawBitmap(originalBitmap, 0, 0, paint);

        return enhancedBitmap;
    }

    /**
     * 裁剪图像
     */
    public static Bitmap cropBitmap(Bitmap source, Rect cropRect) {
        // 验证裁剪矩形
        if (cropRect.left < 0 || cropRect.top < 0 ||
                cropRect.right > source.getWidth() || cropRect.bottom > source.getHeight()) {
            Log.e(TAG, "裁剪区域超出图像范围");
            return source;
        }

        // 执行裁剪
        return Bitmap.createBitmap(
                source,
                cropRect.left,
                cropRect.top,
                cropRect.width(),
                cropRect.height()
        );
    }

    /**
     * 旋转图像
     */
    public static Bitmap rotateBitmap(Bitmap source, float degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    /**
     * 图像锐化
     */
    public static Bitmap sharpenImage(Bitmap original) {
        // 创建源和目标Mat
        Mat src = new Mat();
        Utils.bitmapToMat(original, src);

        // 转换为灰度
        Mat gray = new Mat();
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
        } else {
            gray = src.clone();
        }

        // 应用拉普拉斯滤波器
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, CvType.CV_8U, 3, 1, 0);

        // 添加回原始图像
        Mat sharpened = new Mat();
        Core.addWeighted(gray, 1.5, laplacian, -0.5, 0, sharpened);

        // 转换回Bitmap
        Bitmap result = Bitmap.createBitmap(original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(sharpened, result);

        // 释放资源
        src.release();
        gray.release();
        laplacian.release();
        sharpened.release();

        return result;
    }

    /**
     * 应用透明蒙版
     */
    public static Bitmap applyMask(Bitmap original, Bitmap mask) {
        // 确保大小一致
        if (original.getWidth() != mask.getWidth() || original.getHeight() != mask.getHeight()) {
            mask = Bitmap.createScaledBitmap(mask, original.getWidth(), original.getHeight(), true);
        }

        // 创建结果Bitmap
        Bitmap result = Bitmap.createBitmap(original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);

        // 创建Canvas和Paint
        Canvas canvas = new Canvas(result);
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        // 先绘制原始图像
        canvas.drawBitmap(original, 0, 0, null);

        // 设置混合模式
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));

        // 应用蒙版
        canvas.drawBitmap(mask, 0, 0, paint);

        // 重置混合模式
        paint.setXfermode(null);

        return result;
    }
}