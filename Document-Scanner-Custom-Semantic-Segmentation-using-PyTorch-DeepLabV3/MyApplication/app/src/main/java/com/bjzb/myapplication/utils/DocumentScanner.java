package com.bjzb.myapplication.utils;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
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

    // 处理模型输出并检测文档边缘
    public static Bitmap processModelOutput(Bitmap originalImage, Tensor outputTensor) {
        // 获取模型输出尺寸
        long[] shape = outputTensor.shape();

        // 转换Tensor为OpenCV Mat
        FloatBuffer buffer = FloatBuffer.allocate((int)(shape[1] * shape[2] * shape[3]));
        outputTensor.getDataAsFloatArray();

        // 创建分割掩码
        Mat segmentationMask = new Mat((int)shape[2], (int)shape[3], CvType.CV_32F);
        // 处理输出创建二值掩码 (类似Python代码中的torch.argmax操作)
        processTensorToMask(buffer.array(), segmentationMask, (int)shape[2], (int)shape[3], (int)shape[1]);

        // 转换为8位图像
        Mat mask8U = new Mat();
        segmentationMask.convertTo(mask8U, CvType.CV_8U, 255.0);

        // Canny边缘检测
        Mat edges = new Mat();
        Imgproc.Canny(mask8U, edges, 225, 255);

        // 膨胀边缘
        Mat dilatedEdges = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.dilate(edges, dilatedEdges, kernel);

        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilatedEdges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

        // 如果没有找到轮廓
        if (contours.isEmpty()) {
            Log.e(TAG, "No contours found");
            return originalImage;
        }

        // 按面积排序，获取最大轮廓
        MatOfPoint largestContour = Collections.max(contours, Comparator.comparing(Imgproc::contourArea));

        // 简化轮廓
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        MatOfPoint2f contour2f = new MatOfPoint2f(largestContour.toArray());
        double epsilon = 0.02 * Imgproc.arcLength(contour2f, true);
        Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);

        // 获取角点
        Point[] corners = approxCurve.toArray();
        if (corners.length != 4) {
            Log.w(TAG, "Document corners not found correctly. Found: " + corners.length);
            // 尝试矫正或返回原图
            return originalImage;
        }

        // 调整输出尺寸与原图一致
        float scaleX = (float) originalImage.getWidth() / 384;
        float scaleY = (float) originalImage.getHeight() / 384;

        // 调整角点坐标
        for (int i = 0; i < corners.length; i++) {
            corners[i].x *= scaleX;
            corners[i].y *= scaleY;
        }

        // 对角点排序
        corners = orderPoints(corners);

        // 计算目标角点
        Point[] destCorners = findDestination(corners);

        // 执行透视变换
        return performPerspectiveTransform(originalImage, corners, destCorners);
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

    // 执行透视变换
    private static Bitmap performPerspectiveTransform(Bitmap originalImage, Point[] srcPoints, Point[] dstPoints) {
        // 创建源和目标角点矩阵
        Mat src = new Mat(4, 1, CvType.CV_32FC2);
        Mat dst = new Mat(4, 1, CvType.CV_32FC2);

        for (int i = 0; i < 4; i++) {
            src.put(i, 0, new double[] { srcPoints[i].x, srcPoints[i].y });
            dst.put(i, 0, new double[] { dstPoints[i].x, dstPoints[i].y });
        }

        // 计算透视变换矩阵
        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(src, dst);

        // 创建原始图像的Mat
        Mat inputMat = new Mat();
        Utils.bitmapToMat(originalImage, inputMat);

        // 创建输出Mat
        Mat outputMat = new Mat((int)dstPoints[2].y, (int)dstPoints[2].x, CvType.CV_8UC3);

        // 执行透视变换
        Imgproc.warpPerspective(inputMat, outputMat, perspectiveMatrix, outputMat.size(),
                Imgproc.INTER_LANCZOS4);

        // 转换回Bitmap
        Bitmap resultBitmap = Bitmap.createBitmap(outputMat.cols(), outputMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(outputMat, resultBitmap);

        return resultBitmap;
    }
}