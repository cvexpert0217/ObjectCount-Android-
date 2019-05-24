package org.opencv.samples.facedetect;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;


import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import android.app.Activity;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.LinearLayout;

/*
This is CameraActivity.
 */
public class FdCameraActivity extends Activity implements CvCameraViewListener2 {

    private Mat                    mRgba;
    private Mat                    mGray;
    private DetectionBasedTracker  mNativeDetector;
    private SurfaceViewThread       surfaceViewThread;
    private CameraBridgeViewBase   mOpenCvCameraView;
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
                    mOpenCvCameraView.enableView();

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_camera_surface_view);
        LinearLayout drawTextCanvas = (LinearLayout)findViewById(R.id.drawTextCanvas);
        surfaceViewThread = new SurfaceViewThread(getApplicationContext());
        surfaceViewThread.setMinimumWidth(getWindow().getAttributes().width);
        drawTextCanvas.addView(surfaceViewThread);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        WindowManager windowManager = (WindowManager)getApplicationContext().getSystemService(WINDOW_SERVICE);
        screen_width = windowManager.getDefaultDisplay().getWidth();
        screen_height = windowManager.getDefaultDisplay().getHeight();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
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
        super.onBackPressed();

    }
    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat(height, width, CvType.CV_8UC4);

        mNativeDetector = new DetectionBasedTracker();
        mNativeDetector.startCamera(getIntent().getStringExtra("netPath"),mRgba.width(),mRgba.height(),mRgba.width(),mRgba.height());
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    // input:  Image of Camera
    // output: Image of Detected People
    public Mat onCameraFrame(final CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Mat dst = new Mat();
        Imgproc.cvtColor(mRgba,dst,Imgproc.COLOR_RGBA2RGB);
        mNativeDetector.processCamera(dst , new Callback<byte[]>() {
            @Override
            public void success(final byte[] result) {
                setCameraMat(result, mRgba.width(), mRgba.height());
            }
        });
        return mRgba;
    }

    private void setCameraMat(byte[] result, int width, int height) {
        Mat mat = new Mat(height, width, CvType.CV_8UC4);
        mat.put(0, 0, result);
        mRgba = mat;
    }

}
