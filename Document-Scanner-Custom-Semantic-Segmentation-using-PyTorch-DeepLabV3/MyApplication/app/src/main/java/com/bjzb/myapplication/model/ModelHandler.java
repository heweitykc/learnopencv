package com.bjzb.myapplication.model;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

public class ModelHandler {
    private static final String TAG = "ModelHandler";
    private static final String MODEL_NAME = "doc_scanner_mbv3.pt";

    // 归一化参数 (与Python一致)
    private static final float[] NORM_MEAN = new float[]{0.4611f, 0.4359f, 0.3905f};
    private static final float[] NORM_STD = new float[]{0.2193f, 0.2150f, 0.2109f};

    private Context context;
    private Module model;

    public ModelHandler(Context context) {
        this.context = context;
        try {
            // 从assets加载模型
            model = Module.load(assetFilePath(context, MODEL_NAME));
            Log.i(TAG, "Model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Error loading model: " + e.getMessage());
        }
    }

    // 执行推理
    public Tensor runInference(Bitmap bitmap) {
        if (model == null) {
            Log.e(TAG, "Model not loaded");
            return null;
        }

        // 调整图像大小为模型输入大小
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 384, 384, true);

        // 转换为Tensor
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap, NORM_MEAN, NORM_STD);

        // 执行推理
        Map<String, IValue> outputs = model.forward(IValue.from(inputTensor)).toDictStringKey();
        return outputs.get("out").toTensor();
    }

    // 辅助方法：从assets复制文件到可访问位置
    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}