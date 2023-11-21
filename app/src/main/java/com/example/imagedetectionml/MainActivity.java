package com.example.imagedetectionml;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import com.example.imagedetectionml.databinding.ActivityMainBinding;
import com.example.imagedetectionml.ml.MobilenetV110224Quant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;

    private static final String TAG="MainActivity";
    Bitmap bitmap;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        //permission
        getPermission();
        String[] labels = new String[1001];
        int cnt = 0;
        try{
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line = bufferedReader.readLine();
            while (line!=null){
                Log.d(TAG, "onCreate: "+line);
                labels[cnt++] = line;
                line = bufferedReader.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        binding.selectImage.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });
        binding.captureImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent,101);
            }
        });
        binding.predictImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    bitmap = Bitmap.createScaledBitmap(bitmap,224,224,true);
                    TensorImage image=new TensorImage(DataType.UINT8);
                    image.load(bitmap);
                    ByteBuffer byteBuffer=image.getBuffer();
                    inputFeature0.loadBuffer(byteBuffer);
                    // Runs model inference and gets result.
                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    binding.result.setText("Result:- "+labels[getMax(outputFeature0.getFloatArray())]);



                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    Toast.makeText(MainActivity.this,"Please capture clear image", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                }
            }
        });
    }
    private int getMax(float[] arr){
        int max = 0;
        for(int i=0;i<arr.length;i++){
            if (arr[i]>arr[max]) max = i;
        }
        return max;
    }
    private void getPermission(){
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 11);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == 11){
            if(grantResults.length > 0){
                if(grantResults[0]!=PackageManager.PERMISSION_GRANTED){
                    this.getPermission();
                }
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 100){
            if (data != null){
                Uri uri = data.getData();
                try{
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    binding.imageView.setImageBitmap(bitmap);
                }catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
        else if (requestCode == 101){
            if(data != null){
                bitmap =(Bitmap) data.getExtras().get("data");
                binding.imageView.setImageBitmap(bitmap);
            }


        }
    }
}