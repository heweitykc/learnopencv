package com.bjzb.myapplication;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;


import com.bjzb.myapplication.model.ModelHandler;
import com.bjzb.myapplication.utils.DocumentScanner;
import com.bjzb.myapplication.utils.ImageUtils;

import org.opencv.android.OpenCVLoader;
import org.pytorch.Tensor;

import java.io.FileNotFoundException;
import java.io.OutputStream;

public class ResultActivity extends AppCompatActivity {
    private static final String TAG = "ResultActivity";

    private ImageView ivOriginal;
    private ImageView ivProcessed;
    private ImageView ivMask;
    private ProgressBar progressBar;
    private Button btnSave;

    private Bitmap originalBitmap;
    private Bitmap processedBitmap;
    private ModelHandler modelHandler;

    static {
        if (!OpenCVLoader.initDebug())
            Log.e(TAG, "OpenCV未能初始化！");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        ivOriginal = findViewById(R.id.ivOriginal);
        ivProcessed = findViewById(R.id.ivProcessed);
        ivMask = findViewById(R.id.ivMask);
        progressBar = findViewById(R.id.progressBar);
        btnSave = findViewById(R.id.btnSave);

        modelHandler = new ModelHandler(this);

        // 获取图像来源
        Intent intent = getIntent();
        String source = intent.getStringExtra("source");

        if ("gallery".equals(source)) {
            Uri imageUri = intent.getData();
            if (imageUri != null) {
                try {
                    originalBitmap = ImageUtils.getBitmapFromUri(this, imageUri);
                    processImage();
                } catch (Exception e) {
                    Log.e(TAG, "加载图像出错: " + e.getMessage());
                    Toast.makeText(this, "无法加载图像", Toast.LENGTH_SHORT).show();
                    finish();
                }
            }
        } else if ("camera".equals(source)) {
            byte[] byteArray = intent.getByteArrayExtra("image_data");
            if (byteArray != null) {
                originalBitmap = ImageUtils.getBitmapFromByteArray(byteArray);
                processImage();
            }
        }

        btnSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                saveProcessedImage();
            }
        });
    }

    private void processImage() {
        if (originalBitmap == null) {
            Toast.makeText(this, "无法处理图像", Toast.LENGTH_SHORT).show();
            return;
        }

        ivOriginal.setImageBitmap(originalBitmap);
        progressBar.setVisibility(View.VISIBLE);

        // 在后台线程中处理图像
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // 运行模型推理
                    final Tensor outputTensor = modelHandler.runInference(originalBitmap);
                    
                    // 输出Tensor信息用于调试
                    Log.d(TAG, "模型输出信息: " + modelHandler.printTensorInfo(outputTensor));
                    
                    // 处理输出并获取扫描结果
                    processedBitmap = DocumentScanner.processModelOutput(originalBitmap, outputTensor);

                    // 更新UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            ivProcessed.setImageBitmap(processedBitmap);
                            
                            // 获取并显示掩码图像
                            Bitmap maskBitmap = DocumentScanner.getLastMaskBitmap();
                            if (maskBitmap != null) {
                                ivMask.setVisibility(View.VISIBLE);
                                ivMask.setImageBitmap(maskBitmap);
                            }
                            
                            progressBar.setVisibility(View.GONE);
                            btnSave.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    Log.e(TAG, "处理图像时出错: " + e.getMessage());

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(ResultActivity.this,
                                    "处理图像时出错", Toast.LENGTH_SHORT).show();
                            progressBar.setVisibility(View.GONE);
                        }
                    });
                }
            }
        }).start();
    }

    private void saveProcessedImage() {
        if (processedBitmap == null) {
            Toast.makeText(this, "没有可保存的图像", Toast.LENGTH_SHORT).show();
            return;
        }

        // 创建图像保存路径
        String fileName = "DocScan_" + System.currentTimeMillis() + ".jpg";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

        Uri imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        if (imageUri != null) {
            try {
                OutputStream outputStream = getContentResolver().openOutputStream(imageUri);
                if (outputStream != null) {
                    processedBitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream);
                    outputStream.close();
                    Toast.makeText(this, "图像已保存", Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Log.e(TAG, "保存图像出错: " + e.getMessage());
                Toast.makeText(this, "保存图像失败", Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this, "无法创建图像文件", Toast.LENGTH_SHORT).show();
        }
    }
}