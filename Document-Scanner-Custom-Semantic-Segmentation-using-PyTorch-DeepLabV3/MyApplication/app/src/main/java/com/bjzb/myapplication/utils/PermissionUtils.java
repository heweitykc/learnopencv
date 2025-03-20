package com.bjzb.myapplication.utils;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import java.util.ArrayList;
import java.util.List;

/**
 * Android 权限管理工具类
 * 用于处理运行时权限请求和结果回调
 */
public class PermissionUtils {
    private static final String TAG = "PermissionUtils";

    // 常用权限常量
    public static final String CAMERA = Manifest.permission.CAMERA;
    public static final String READ_EXTERNAL_STORAGE = Manifest.permission.READ_EXTERNAL_STORAGE;
    public static final String WRITE_EXTERNAL_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    // 权限请求接口
    public interface PermissionCallback {
        void onPermissionGranted();
        void onPermissionDenied(List<String> deniedPermissions);
    }

    /**
     * 检查单个权限是否已授予
     */
    public static boolean hasPermission(Context context, String permission) {
        return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * 检查多个权限是否已全部授予
     */
    public static boolean hasPermissions(Context context, String... permissions) {
        for (String permission : permissions) {
            if (!hasPermission(context, permission)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 请求单个权限
     */
    public static void requestPermission(Activity activity, String permission, int requestCode) {
        ActivityCompat.requestPermissions(activity, new String[]{permission}, requestCode);
    }

    /**
     * 请求单个权限（使用Fragment）
     */
    public static void requestPermission(Fragment fragment, String permission, int requestCode) {
        fragment.requestPermissions(new String[]{permission}, requestCode);
    }

    /**
     * 请求多个权限
     */
    public static void requestPermissions(Activity activity, String[] permissions, int requestCode) {
        ActivityCompat.requestPermissions(activity, permissions, requestCode);
    }

    /**
     * 请求多个权限（使用Fragment）
     */
    public static void requestPermissions(Fragment fragment, String[] permissions, int requestCode) {
        fragment.requestPermissions(permissions, requestCode);
    }

    /**
     * 请求相机和存储权限（常用组合）
     */
    public static void requestCameraAndStoragePermissions(Activity activity, int requestCode) {
        String[] permissions = {CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE};
        requestPermissions(activity, permissions, requestCode);
    }

    /**
     * 使用回调方式请求权限
     */
    public static void requestPermissionsWithCallback(final Activity activity, final String[] permissions,
                                                      final int requestCode, final PermissionCallback callback) {
        // 检查是否已拥有所有权限
        if (hasPermissions(activity, permissions)) {
            // 已有所有权限，直接回调授权成功
            callback.onPermissionGranted();
            return;
        }

        // 记录已拒绝的权限
        final List<String> deniedPermissions = new ArrayList<>();
        for (String permission : permissions) {
            if (!hasPermission(activity, permission)) {
                deniedPermissions.add(permission);
            }
        }

        // 请求尚未授予的权限
        String[] permissionsToRequest = deniedPermissions.toArray(new String[0]);

        // 设置权限结果处理器
        PermissionResultHandler.registerCallback(requestCode, new PermissionResultHandler.PermissionResultCallback() {
            @Override
            public void onPermissionResult(int requestCode, String[] permissions, int[] grantResults) {
                List<String> deniedList = new ArrayList<>();
                for (int i = 0; i < permissions.length; i++) {
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                        deniedList.add(permissions[i]);
                    }
                }

                if (deniedList.isEmpty()) {
                    // 所有权限都获得授权
                    callback.onPermissionGranted();
                } else {
                    // 有权限被拒绝
                    callback.onPermissionDenied(deniedList);
                }

                // 请求处理完毕，注销回调
                PermissionResultHandler.unregisterCallback(requestCode);
            }
        });

        // 发起权限请求
        requestPermissions(activity, permissionsToRequest, requestCode);
    }

    /**
     * 检查是否用户选择了"不再询问"
     */
    public static boolean shouldShowRationale(Activity activity, String permission) {
        return ActivityCompat.shouldShowRequestPermissionRationale(activity, permission);
    }

    /**
     * 检查是否用户选择了"不再询问"（使用Fragment）
     */
    public static boolean shouldShowRationale(Fragment fragment, String permission) {
        return fragment.shouldShowRequestPermissionRationale(permission);
    }

    /**
     * 显示权限说明对话框
     */
    public static void showPermissionRationaleDialog(Activity activity, String title, String message,
                                                     final String[] permissions, final int requestCode) {
        new AlertDialog.Builder(activity)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        requestPermissions(activity, permissions, requestCode);
                    }
                })
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .create()
                .show();
    }

    /**
     * 显示打开应用设置界面的对话框
     * 当用户选择了"不再询问"后，需要引导用户手动开启权限
     */
    public static void showOpenSettingsDialog(final Activity activity, String title, String message) {
        new AlertDialog.Builder(activity)
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton("去设置", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        openAppSettings(activity);
                    }
                })
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .create()
                .show();
    }

    /**
     * 打开应用设置界面
     */
    public static void openAppSettings(Activity activity) {
        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
        Uri uri = Uri.fromParts("package", activity.getPackageName(), null);
        intent.setData(uri);
        activity.startActivity(intent);
    }

    /**
     * 处理权限请求结果
     * 在Activity或Fragment的onRequestPermissionsResult方法中调用
     */
    public static boolean handlePermissionResult(int requestCode, @NonNull String[] permissions,
                                                 @NonNull int[] grantResults) {
        // 转发给结果处理器
        return PermissionResultHandler.handlePermissionResult(requestCode, permissions, grantResults);
    }

    /**
     * 权限结果处理器，管理权限请求回调
     */
    private static class PermissionResultHandler {
        // 回调接口
        interface PermissionResultCallback {
            void onPermissionResult(int requestCode, String[] permissions, int[] grantResults);
        }

        // 存储请求码和对应的回调
        private static final android.util.SparseArray<PermissionResultCallback> callbackMap = new android.util.SparseArray<>();

        // 注册回调
        static void registerCallback(int requestCode, PermissionResultCallback callback) {
            callbackMap.put(requestCode, callback);
        }

        // 注销回调
        static void unregisterCallback(int requestCode) {
            callbackMap.remove(requestCode);
        }

        // 处理权限请求结果
        static boolean handlePermissionResult(int requestCode, @NonNull String[] permissions,
                                              @NonNull int[] grantResults) {
            PermissionResultCallback callback = callbackMap.get(requestCode);
            if (callback != null) {
                callback.onPermissionResult(requestCode, permissions, grantResults);
                return true;
            }
            return false;
        }
    }

    /**
     * 简单检查存储权限
     */
    public static boolean checkStoragePermission(Activity activity, int requestCode) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+, 使用新的存储权限机制
            return true; // 在Android 11+上可以使用MediaStore API无需传统存储权限
        } else {
            // Android 10及以下使用传统存储权限
            if (!hasPermission(activity, WRITE_EXTERNAL_STORAGE)) {
                requestPermission(activity, WRITE_EXTERNAL_STORAGE, requestCode);
                return false;
            }
            return true;
        }
    }

    /**
     * 简单检查相机权限
     */
    public static boolean checkCameraPermission(Activity activity, int requestCode) {
        if (!hasPermission(activity, CAMERA)) {
            requestPermission(activity, CAMERA, requestCode);
            return false;
        }
        return true;
    }

    /**
     * 根据权限类型获取友好的权限名称
     */
    public static String getPermissionFriendlyName(String permission) {
        switch (permission) {
            case CAMERA:
                return "相机";
            case READ_EXTERNAL_STORAGE:
            case WRITE_EXTERNAL_STORAGE:
                return "存储";
            default:
                return permission;
        }
    }
}