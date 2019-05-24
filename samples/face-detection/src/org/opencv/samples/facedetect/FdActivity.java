package org.opencv.samples.facedetect;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;


import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PixelFormat;
import android.media.MediaMetadataRetriever;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.LinearLayout;


/*
This is Activity for Image and Video.
 */
public class FdActivity extends Activity {

    private DetectionBasedTracker   mNativeDetector;
    private SurfaceViewThread       surfaceViewThread;
    private boolean                 bDetectStatus;
    private int                     screen_width;
    private int                     screen_height;
    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (bDetectStatus){
                super.onManagerConnected(status);
                return;
            }
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    bDetectStatus = true;

                    System.loadLibrary("detection_based_tracker");

                    Runnable r = new Runnable() {
                        @Override
                        public void run() {
                            mNativeDetector = new DetectionBasedTracker();
                            int RealWidth = 0;
                            int RealHeight = 0;
                            if (getIntent().getIntExtra("type", -1) == 0) {
                                MediaMetadataRetriever retriever = new MediaMetadataRetriever();
                                retriever.setDataSource(getIntent().getStringExtra("path"));
                                RealWidth = Integer.valueOf(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
                                RealHeight = Integer.valueOf(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
                                retriever.release();
                            } else if (getIntent().getIntExtra("type", -1) == 1) {
                                Bitmap bitmap = BitmapFactory.decodeFile(getIntent().getStringExtra("path"));
                                RealHeight = bitmap.getHeight();
                                RealWidth = bitmap.getWidth();
                            }
                            mNativeDetector.start(getIntent().getStringExtra("netPath"),RealWidth,RealHeight, screen_width, screen_height, getIntent().getIntExtra("type", -1), getIntent().getStringExtra("path"), new Callback<byte[]>() {
                                @Override
                                public void success(byte[] result) {
                                    surfaceViewThread.setImage(result, screen_width, screen_height);
                                }
                            });
                        }
                    };
                    new Thread(r).start();

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Called when the activity is first created. */
    @SuppressLint("WrongViewCast")
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        WindowManager windowManager = (WindowManager)getApplicationContext().getSystemService(WINDOW_SERVICE);
        screen_width = windowManager.getDefaultDisplay().getWidth();
        screen_height = windowManager.getDefaultDisplay().getHeight();
        setContentView(R.layout.face_detect_surface_view);
        LinearLayout drawTextCanvas = (LinearLayout)findViewById(R.id.drawTextCanvas);
        surfaceViewThread = new SurfaceViewThread(getApplicationContext());
        surfaceViewThread.setMinimumWidth(getWindow().getAttributes().width);
        surfaceViewThread.setZOrderOnTop(true);

        surfaceViewThread.getHolder().setFormat(PixelFormat.RGBA_8888);
        drawTextCanvas.addView(surfaceViewThread);
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onBackPressed() {
        surfaceViewThread.threadRunning = false;
        bDetectStatus = false;
        mNativeDetector.stop(getIntent().getIntExtra("type", -1));
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        super.onBackPressed();
    }

    public void onDestroy() {
        super.onDestroy();
    }

}
