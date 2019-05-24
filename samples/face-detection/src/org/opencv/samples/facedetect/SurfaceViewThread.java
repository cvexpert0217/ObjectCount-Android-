package org.opencv.samples.facedetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Handler;
import android.text.TextUtils;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.locks.ReentrantLock;

// This is Viewer class for detecting image.
public class SurfaceViewThread extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private DetectionBasedTracker  mNativeDetector;


    private SurfaceHolder surfaceHolder = null;

    private Paint paint = null;

    private Thread thread = null;
    private byte[] data;
    private Bitmap g_bitmap;

    // Record whether the child thread is running or not.
    public static boolean threadRunning = false;

    private Canvas canvas = null;
    private int width = 0;
    private int height = 0;

    ReentrantLock threadLock = new ReentrantLock();

    private static String LOG_TAG = "SURFACE_VIEW_THREAD";

    public SurfaceViewThread(Context context) {
        super(context);

        setFocusable(true);

        // Get SurfaceHolder object.
        surfaceHolder = this.getHolder();
        // Add current object as the callback listener.
        surfaceHolder.addCallback(this);

        // Create the paint object which will draw the text.
        paint = new Paint();
        paint.setTextSize(100);
        paint.setColor(Color.GREEN);

        // Set the SurfaceView object at the top of View object.
        setZOrderOnTop(true);

        //setBackgroundColor(Color.RED);
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {

        // Create the child thread when SurfaceView is created.
        thread = new Thread(this);
        // Start to run the child thread.
        thread.start();
        // Set thread running flag to true.
        threadRunning = true;

    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        // Set thread running flag to false when Surface is destroyed.
        // Then the thread will jump out the while loop and complete.
        threadRunning = false;
    }

    @Override
    public void run() {
        Log.i("debug", "surfaceview run");
        while(threadRunning)
        {
            drawImage();
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void drawImage() {
        if (g_bitmap == null)
            return;

        canvas = surfaceHolder.lockCanvas();
        canvas.drawBitmap(g_bitmap,0,0, null);
        surfaceHolder.unlockCanvasAndPost(canvas);
    }

    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }
    public void setImage(byte[] data, int width, int height) {
        this.width = width;
        this.height = height;
        this.data = data;
        if (data == null)
        {
            Log.i("debug", "Data is Null.");
            return;
        }
        Log.i("debug", "Data is NOT Null.");
        Mat mat = new Mat(height, width, CvType.CV_8UC4);
        mat.put(0, 0, data);

        Bitmap bitmap = convertMatToBitMap(mat);

        if(bitmap == null) {
            return;
        }
        g_bitmap = bitmap;
    }
}