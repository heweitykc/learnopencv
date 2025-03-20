package com.bjzb.myapplication.utils;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 文档边缘检测工具类
 * 用于在图像中检测文档轮廓并提供边缘坐标
 */
public class DocumentDetector {
    private static final String TAG = "DocumentDetector";
    private static final int DOWNSCALE_IMAGE_SIZE = 600; // 处理图像的最大尺寸

    /**
     * 实时检测文档边缘
     * @param bitmap 相机预览图像
     * @return 文档四个角点的坐标，如果未检测到则返回null
     */
    public static org.opencv.core.Point[] detectDocumentEdges(Bitmap bitmap) {
        // 转换为OpenCV Mat
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);

        // 调整图像大小以加快处理速度
        Mat resizedMat = resizeImage(srcMat, DOWNSCALE_IMAGE_SIZE);

        // 转换为灰度图
        Mat grayMat = new Mat();
        Imgproc.cvtColor(resizedMat, grayMat, Imgproc.COLOR_BGR2GRAY);

        // 应用高斯模糊减少噪声
        Imgproc.GaussianBlur(grayMat, grayMat, new Size(5, 5), 0);

        // 边缘检测
        Mat edges = new Mat();
        Imgproc.Canny(grayMat, edges, 75, 200);

        // 膨胀边缘
        Mat dilatedEdges = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.dilate(edges, dilatedEdges, kernel);

        // 查找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilatedEdges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // 如果没有找到轮廓，返回null
        if (contours.isEmpty()) {
            Log.d(TAG, "No contours found 1");
            return null;
        }

        // 按面积排序，获取最大的几个轮廓
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return Double.compare(Imgproc.contourArea(o2), Imgproc.contourArea(o1));
            }
        });

        // 遍历轮廓寻找最可能的文档轮廓
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            // 忽略太小的轮廓
            if (area < 0.05 * resizedMat.width() * resizedMat.height()) {
                continue;
            }

            // 轮廓近似
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double epsilon = 0.02 * Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);

            // 如果近似后的轮廓有4个点，则可能是文档
            if (approxCurve.total() == 4) {
                org.opencv.core.Point[] points = approxCurve.toArray();

                // 验证是否是凸四边形
                if (isConvexQuadrilateral(points)) {
                    // 调整回原始图像大小
                    double scaleX = (double) srcMat.width() / resizedMat.width();
                    double scaleY = (double) srcMat.height() / resizedMat.height();

                    for (int i = 0; i < points.length; i++) {
                        points[i].x *= scaleX;
                        points[i].y *= scaleY;
                    }

                    // 对坐标点进行排序 (左上, 右上, 右下, 左下)
                    return sortPoints(points);
                }
            }
        }

        // 未找到合适的文档轮廓
        return null;
    }

    /**
     * 使用预处理后的二值掩码图像检测文档边缘
     * @param binaryMask 通过模型处理后的二值掩码
     * @param originalWidth 原始图像宽度
     * @param originalHeight 原始图像高度
     * @return 文档四个角点的坐标，如果未检测到则返回null
     */
    public static org.opencv.core.Point[] detectDocumentFromMask(Mat binaryMask, int originalWidth, int originalHeight) {
        // 确保掩码是二值图像
        Mat mask = new Mat();
        if (binaryMask.type() != CvType.CV_8UC1) {
            binaryMask.convertTo(mask, CvType.CV_8UC1, 255.0);
        } else {
            mask = binaryMask.clone();
        }

        // 寻找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            Log.d(TAG, "No contours found in mask");
            return null;
        }

        // 找到最大轮廓
        MatOfPoint largestContour = Collections.max(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return Double.compare(Imgproc.contourArea(o1), Imgproc.contourArea(o2));
            }
        });

        // 轮廓近似
        MatOfPoint2f contour2f = new MatOfPoint2f(largestContour.toArray());
        double epsilon = 0.02 * Imgproc.arcLength(contour2f, true);
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approxCurve, epsilon, true);

        org.opencv.core.Point[] points;

        // 如果近似后的轮廓有4个点，直接使用
        if (approxCurve.total() == 4) {
            points = approxCurve.toArray();
        } else {
            // 如果点数不是4，找到最小外接矩形
            org.opencv.core.RotatedRect rect = Imgproc.minAreaRect(contour2f);
            org.opencv.core.Point[] rectPoints = new org.opencv.core.Point[4];
            rect.points(rectPoints);
            points = rectPoints;
        }

        // 调整回原始图像大小
        double scaleX = (double) originalWidth / mask.width();
        double scaleY = (double) originalHeight / mask.height();

        for (int i = 0; i < points.length; i++) {
            points[i].x *= scaleX;
            points[i].y *= scaleY;
        }

        // 排序角点
        return sortPoints(points);
    }

    /**
     * 调整图像大小，保持比例
     */
    private static Mat resizeImage(Mat src, int maxSize) {
        double ratio = Math.min(maxSize / (double) src.width(), maxSize / (double) src.height());
        Size newSize = new Size(src.width() * ratio, src.height() * ratio);

        Mat resized = new Mat();
        Imgproc.resize(src, resized, newSize);
        return resized;
    }

    /**
     * 检查是否为凸四边形
     */
    private static boolean isConvexQuadrilateral(org.opencv.core.Point[] points) {
        if (points.length != 4) return false;

        // 创建点集合
        MatOfPoint2f contour = new MatOfPoint2f(points);
        // 检查凸性
        return Imgproc.isContourConvex(new MatOfPoint(points));
    }

    /**
     * 对点进行排序：左上、右上、右下、左下
     */
    private static org.opencv.core.Point[] sortPoints(org.opencv.core.Point[] pts) {
        org.opencv.core.Point[] sortedPoints = new org.opencv.core.Point[4];

        // 计算质心
        double centerX = 0, centerY = 0;
        for (org.opencv.core.Point pt : pts) {
            centerX += pt.x;
            centerY += pt.y;
        }
        centerX /= pts.length;
        centerY /= pts.length;

        // 左上角和右下角点
        ArrayList<org.opencv.core.Point> topLeft = new ArrayList<>();
        ArrayList<org.opencv.core.Point> bottomRight = new ArrayList<>();

        // 分类点
        for (org.opencv.core.Point pt : pts) {
            if (pt.x < centerX && pt.y < centerY) {
                topLeft.add(pt); // 左上象限
            } else if (pt.x > centerX && pt.y > centerY) {
                bottomRight.add(pt); // 右下象限
            } else if (pt.x < centerX && pt.y > centerY) {
                sortedPoints[3] = pt; // 左下
            } else {
                sortedPoints[1] = pt; // 右上
            }
        }

        // 如果有多个点在左上象限，选择x+y最小的
        if (!topLeft.isEmpty()) {
            sortedPoints[0] = Collections.min(topLeft, new Comparator<org.opencv.core.Point>() {
                @Override
                public int compare(org.opencv.core.Point o1, org.opencv.core.Point o2) {
                    return Double.compare(o1.x + o1.y, o2.x + o2.y);
                }
            });
        }

        // 如果有多个点在右下象限，选择x+y最大的
        if (!bottomRight.isEmpty()) {
            sortedPoints[2] = Collections.max(bottomRight, new Comparator<org.opencv.core.Point>() {
                @Override
                public int compare(org.opencv.core.Point o1, org.opencv.core.Point o2) {
                    return Double.compare(o1.x + o1.y, o2.x + o2.y);
                }
            });
        }

        return sortedPoints;
    }

    /**
     * 绘制文档边缘
     * @param bitmap 原始图像
     * @param points 文档四个角点
     * @return 带有边缘标记的图像
     */
    public static Bitmap drawDocumentEdges(Bitmap bitmap, org.opencv.core.Point[] points) {
        if (points == null || points.length != 4) {
            return bitmap;
        }

        // 转换为OpenCV Mat
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        // 绘制边缘
        for (int i = 0; i < 4; i++) {
            Imgproc.line(
                    mat,
                    points[i],
                    points[(i + 1) % 4],
                    new org.opencv.core.Scalar(0, 255, 0), // 绿色
                    3 // 线宽
            );
        }

        // 转换回Bitmap
        Bitmap result = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
        Utils.matToBitmap(mat, result);
        return result;
    }
}