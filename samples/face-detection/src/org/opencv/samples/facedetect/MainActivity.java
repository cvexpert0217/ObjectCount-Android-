package org.opencv.samples.facedetect;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.Toast;

import java.io.File;

import java.io.IOException;

import static android.os.Environment.getExternalStorageDirectory;

/*
This is start Activity.
 */
public class MainActivity extends Activity {
    static int index = 0;
    final static int VIDEO_SELECT_CODE = 3000;
    final static int IMAGE_SELECT_CODE = 3001;
    boolean permission_result = true;
    Button mVideoButton;
    Button mImageButton;
    Button mCameraButton;
    Button mExitButton;
    RelativeLayout progressLayout;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.layout_main);
        mVideoButton = (Button)findViewById(R.id.video);
        mImageButton= (Button)findViewById(R.id.image);
        mCameraButton = (Button)findViewById(R.id.camera);
        mExitButton = (Button)findViewById(R.id.exit);
        progressLayout = (RelativeLayout)findViewById(R.id.progressLayout);

        mVideoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showFileChooser("video");
            }
        });

        mImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showFileChooser("image");
            }
        });

        mCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showCamera();
            }
        });

        mExitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
            }
        });

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE ) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE ) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA ) == PackageManager.PERMISSION_GRANTED) {

            Runnable r = new Runnable() {
                @Override
                public void run() {
                    CheckRawFiles();
                }
            };
            Thread t = new Thread(r);
            t.start();
        } else {
            SetPermission(4000);
        }
    }

    //Callback function of Permission dialog.
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case 4000: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED
                        && grantResults[2] == PackageManager.PERMISSION_GRANTED) {

                    Runnable r = new Runnable() {
                        @Override
                        public void run() {
                            CheckRawFiles();
                        }
                    };
                    Thread t = new Thread(r);
                    t.start();
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {
                    finish();
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }

        }
    }

    //Function of setting the needed permission. (permission = {Read_external_storage, Write_external_storage, Camera}
    private void SetPermission( int permission_Number) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.CAMERA}, permission_Number);
        }
    }

    // Function about choose types of files.
    private void showFileChooser(String selectType) {

        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType(selectType + "/*" );
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        try {
            startActivityForResult(
                    Intent.createChooser(intent, "Select a File to Upload"),
                    selectType == "video" ? VIDEO_SELECT_CODE : IMAGE_SELECT_CODE);
        } catch (android.content.ActivityNotFoundException ex) {
        }

    }
    private void showCamera() {
        Intent intent = new Intent(this,FdCameraActivity.class);
        intent.putExtra("type", 2);
        intent.putExtra("netPath", getExternalStorageDirectory() + "/Download");
        startActivity(intent);
    }
    private void showDetection(int type, String path) {
        Intent intent = new Intent(this, FdActivity.class);
        intent.putExtra("type", type);
        intent.putExtra("path", path);
        intent.putExtra("netPath", getExternalStorageDirectory() + "/Download");
        startActivity(intent);
    }

    //Function of checking obb file exists and unzip it on Download directory.
    private void CheckRawFiles() {
        boolean unzipFlag = false;
        String rawData[] = {"net.cfg","net.wts","net.nms"};

        for (int i = 0 ; i < rawData.length; i ++) {
            File file = new File(getExternalStorageDirectory() + "/Download/" + rawData[i]);
            if (!file.exists()) {
                unzipFlag = true;
            }
        }
        if (!unzipFlag) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    progressLayout.setVisibility(View.INVISIBLE);
                }
            });
            return;
        }
        if (unzipFile() == true) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    progressLayout.setVisibility(View.INVISIBLE);
                }
            });

            return;
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(getApplicationContext(),"Can't Find OBB File in Right Place", Toast.LENGTH_LONG).show();
            }
        });
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        finish();
    }

    public boolean unzipFile() {
        boolean result = false;
        ZipResourceFile expansionFile = null;
        try {
            expansionFile = APKExpansionSupport
                    .getAPKExpansionZipFile(this, 3, 0);

        } catch (IOException e1) {
            e1.printStackTrace();

        }
        if (expansionFile == null) {
            result = false;
        } else {
            ZipResourceFile.ZipEntryRO[] zip = null;
            try {
                zip = expansionFile.getAllEntries();
            } catch (Exception e) {
                e.printStackTrace();
            }

            File file = new File(Environment.getExternalStorageDirectory()
                    .getAbsolutePath() + "/Download");
            ZipHelper.unzip(zip[0].mZipFileName, file);
            if (file.exists()) {
                Log.e("", "unzipped : " + file.getAbsolutePath());
            }
            result = true;
        }
        return  result;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        int type = -1;
        switch (requestCode) {
            case VIDEO_SELECT_CODE:
                type = 0;
                break;
            case IMAGE_SELECT_CODE:
                type = 1;
                break;
        }

        if (resultCode == RESULT_OK) {
            Uri uri = data.getData();
            String path = RealPathUtil.getRealPath(getApplicationContext(), uri);
            showDetection(type, path);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    @Override
    protected void onResume() {
        super.onResume();
    }
}
