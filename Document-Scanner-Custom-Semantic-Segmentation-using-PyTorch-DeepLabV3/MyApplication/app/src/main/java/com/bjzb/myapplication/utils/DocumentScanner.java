package com.bjzb.myapplication.utils;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.pytorch.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class DocumentScanner {
    private static final String TAG = "DocumentScanner";

    // 添加一个静态变量存储最后一次处理的掩码
    private static Bitmap lastMaskBitmap = null;

    // 添加getter方法
    public static Bitmap getLastMaskBitmap() {
        return lastMaskBitmap;
    }

    // 处理模型输出并检测文档边缘
    public static Bitmap processModelOutput(Bitmap originalImage, Tensor outputTensor) {
        try {
            // 获取模型输出尺寸
            long[] shape = outputTensor.shape();
            Log.d(TAG, "模型输出尺寸: " + Arrays.toString(shape));
            
            // 转换Tensor为OpenCV Mat
            float[] dataArray = outputTensor.getDataAsFloatArray();
            Log.d(TAG, "输出数据大小: " + dataArray.length);
            
            // 根据shape分析输出结构
            int maskHeight, maskWidth;
            if (shape.length == 4) { // 通常是[1, C, H, W]或[1, H, W, C]格式
                // 假设格式为[1, C, H, W]
                maskHeight = (int)shape[2];
                maskWidth = (int)shape[3];
            } else if (shape.length == 3) { // 可能是[C, H, W]
                maskHeight = (int)shape[1];
                maskWidth = (int)shape[2];
            } else {
                Log.e(TAG, "无法理解的模型输出形状: " + Arrays.toString(shape));
                return originalImage;
            }
            
            Log.d(TAG, "掩码尺寸: " + maskHeight + "x" + maskWidth);
            
            // 创建分割掩码
            Mat segmentationMask = new Mat(maskHeight, maskWidth, CvType.CV_32F);
            
            // 改用简单的阈值方法处理 - 直接使用第一个通道
            // 对于二分类问题，可以假设第一个通道的值越大越可能是背景，越小越可能是文档
            for (int y = 0; y < maskHeight; y++) {
                for (int x = 0; x < maskWidth; x++) {
                    int idx = y * maskWidth + x;
                    // 如果只有一个通道，直接用它
                    if (dataArray.length == maskHeight * maskWidth) {
                        float value = dataArray[idx];
                        // 使用阈值0.5（如果模型输出是sigmoid/softmax的话）
                        segmentationMask.put(y, x, value > 0.5f ? 1.0f : 0.0f);
                    }
                    // 如果有两个通道，比较它们
                    else if (dataArray.length >= maskHeight * maskWidth * 2) {
                        try {
                            float val0 = dataArray[idx]; // 第一个通道
                            float val1 = dataArray[maskHeight * maskWidth + idx]; // 第二个通道
                            float value = (val1 > val0) ? 1.0f : 0.0f;
                            segmentationMask.put(y, x, value);
                        } catch (IndexOutOfBoundsException e) {
                            // 索引错误，输出更多调试信息
                            Log.e(TAG, "索引越界: " + e.getMessage() + 
                                  ", 位置["+y+","+x+"], 总大小="+dataArray.length);
                            return originalImage;
                        }
                    }
                    // 否则无法处理
                    else {
                        Log.e(TAG, "未知的输出格式: 数据长度=" + dataArray.length + 
                              " 但掩码大小=" + (maskHeight * maskWidth));
                        return originalImage;
                    }
                }
            }
            
            // 添加调试信息
            Log.d(TAG, "掩码值范围: " + Core.minMaxLoc(segmentationMask).minVal +
                  " 到 " + Core.minMaxLoc(segmentationMask).maxVal);
            
            // 转换为8位图像
            Mat mask8U = new Mat();
            segmentationMask.convertTo(mask8U, CvType.CV_8U, 255.0);
            
            // 保存中间结果用于调试
            Bitmap maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mask8U, maskBitmap);
            lastMaskBitmap = maskBitmap; // 假设您添加了这个静态字段
            
            // 应用膨胀操作改善掩码
            Mat dilatedMask = new Mat();
            Mat dilationKernel = Imgproc.getStructuringElement(
                    Imgproc.MORPH_ELLIPSE, new Size(5, 5));
            Imgproc.dilate(mask8U, dilatedMask, dilationKernel);
            
            // 使用DocumentDetector中的方法来处理掩码
            Point[] documentCorners = DocumentDetector.detectDocumentFromMask(
                    dilatedMask, originalImage.getWidth(), originalImage.getHeight());
            
            // 如果检测到文档角点，应用透视变换
            if (documentCorners != null && documentCorners.length == 4) {
                return applyPerspectiveTransform(originalImage, documentCorners);
            } else {
                Log.e(TAG, "未能检测到文档角点");
                return originalImage;
            }
            
        } catch (Exception e) {
            Log.e(TAG, "文档扫描错误: " + e.getMessage());
            e.printStackTrace();
            return originalImage;
        }
    }

    // 以下是辅助方法，实现与Python代码相同的功能

    // 处理输出Tensor创建分割掩码
    private static void processTensorToMask(float[] data, Mat mask, int height, int width, int channels) {
        // 实现argmax操作
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxClass = 0;

                for (int c = 0; c < channels; c++) {
                    int index = (c * height * width) + (y * width) + x;
                    if (data[index] > maxVal) {
                        maxVal = data[index];
                        maxClass = c;
                    }
                }

                // 将非零类别设为1
                mask.put(y, x, maxClass > 0 ? 1.0f : 0.0f);
            }
        }
    }

    // 角点排序 (类似Python的order_points函数)
    private static Point[] orderPoints(Point[] pts) {
        Point[] rect = new Point[4];

        // 计算每个点的坐标和
        double[] sums = new double[pts.length];
        for (int i = 0; i < pts.length; i++) {
            sums[i] = pts[i].x + pts[i].y;
        }

        // 左上角点(坐标和最小)
        int topLeftIdx = findMinIndex(sums);
        rect[0] = pts[topLeftIdx];

        // 右下角点(坐标和最大)
        int bottomRightIdx = findMaxIndex(sums);
        rect[2] = pts[bottomRightIdx];

        // 计算每个点的坐标差
        double[] diffs = new double[pts.length];
        for (int i = 0; i < pts.length; i++) {
            diffs[i] = pts[i].y - pts[i].x;
        }

        // 右上角点(坐标差最小)
        int topRightIdx = findMinIndex(diffs);
        rect[1] = pts[topRightIdx];

        // 左下角点(坐标差最大)
        int bottomLeftIdx = findMaxIndex(diffs);
        rect[3] = pts[bottomLeftIdx];

        return rect;
    }

    // 查找最小值索引
    private static int findMinIndex(double[] arr) {
        int minIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[minIdx]) minIdx = i;
        }
        return minIdx;
    }

    // 查找最大值索引
    private static int findMaxIndex(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    // 计算目标角点 (类似Python的find_dest函数)
    private static Point[] findDestination(Point[] pts) {
        Point tl = pts[0];
        Point tr = pts[1];
        Point br = pts[2];
        Point bl = pts[3];

        // 计算最大宽度
        double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
        double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));
        int maxWidth = (int)Math.max(widthA, widthB);

        // 计算最大高度
        double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
        double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));
        int maxHeight = (int)Math.max(heightA, heightB);

        // 创建目标角点
        Point[] dst = new Point[4];
        dst[0] = new Point(0, 0);
        dst[1] = new Point(maxWidth - 1, 0);
        dst[2] = new Point(maxWidth - 1, maxHeight - 1);
        dst[3] = new Point(0, maxHeight - 1);

        return dst;
    }

    /**
     * 执行透视变换
     * @param originalImage 原始图像
     * @param corners 文档四个角点
     * @return 透视校正后的图像
     */
    private static Bitmap applyPerspectiveTransform(Bitmap originalImage, Point[] corners) {
        try {
            // 确保有四个角点
            if (corners == null || corners.length != 4) {
                Log.e(TAG, "透视变换需要四个角点");
                return originalImage;
            }

            // 将角点排序为: 左上, 右上, 右下, 左下
            corners = orderPoints(corners);

            // 计算目标尺寸
            double width1 = Math.sqrt(Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2));
            double width2 = Math.sqrt(Math.pow(corners[2].x - corners[3].x, 2) + Math.pow(corners[2].y - corners[3].y, 2));
            double maxWidth = Math.max(width1, width2);

            double height1 = Math.sqrt(Math.pow(corners[3].x - corners[0].x, 2) + Math.pow(corners[3].y - corners[0].y, 2));
            double height2 = Math.sqrt(Math.pow(corners[2].x - corners[1].x, 2) + Math.pow(corners[2].y - corners[1].y, 2));
            double maxHeight = Math.max(height1, height2);

            // 创建目标角点
            MatOfPoint2f dstPoints = new MatOfPoint2f(
                    new Point(0, 0),                   // 左上
                    new Point(maxWidth - 1, 0),        // 右上
                    new Point(maxWidth - 1, maxHeight - 1), // 右下
                    new Point(0, maxHeight - 1)        // 左下
            );

            // 创建源角点
            MatOfPoint2f srcPoints = new MatOfPoint2f(corners);

            // 计算透视变换矩阵
            Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);

            // 将原始图像转换为Mat
            Mat srcMat = new Mat();
            Utils.bitmapToMat(originalImage, srcMat);

            // 创建目标Mat
            Mat dstMat = new Mat((int) maxHeight, (int) maxWidth, srcMat.type());

            // 执行透视变换
            Imgproc.warpPerspective(srcMat, dstMat, perspectiveMatrix, dstMat.size());

            // 将结果转换回Bitmap
            Bitmap resultBitmap = Bitmap.createBitmap((int) maxWidth, (int) maxHeight, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dstMat, resultBitmap);

            // 释放资源
            srcMat.release();
            dstMat.release();
            perspectiveMatrix.release();

            return resultBitmap;
        } catch (Exception e) {
            Log.e(TAG, "透视变换错误: " + e.getMessage());
            e.printStackTrace();
            return originalImage;
        }
    }

    /**
     * 获取数组最小值索引
     */
    private static int minIndex(double[] arr) {
        int minIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < arr[minIdx]) {
                minIdx = i;
            }
        }
        return minIdx;
    }

    /**
     * 获取数组最大值索引
     */
    private static int maxIndex(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}