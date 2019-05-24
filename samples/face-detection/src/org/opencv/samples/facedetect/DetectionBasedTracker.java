package org.opencv.samples.facedetect;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;



public class DetectionBasedTracker
{
    public DetectionBasedTracker() {

    }
    public DetectionBasedTracker(String cascadeName, int minFaceSize) {
        mNativeObj = nativeCreateObject(cascadeName, minFaceSize);
    }

    public void start(String netPath,int RealWidth, int RealHeight,int width, int height, int type, String fileName, Callback<byte[]> callback) {
        nativeStart(netPath,RealWidth,RealHeight, width, height, type, fileName, callback);
    }

    public void startCamera(String netPath, int RealWidth, int RealHeight, int width, int height) {
        nativeStartCamera(netPath, RealWidth, RealHeight,width, height);
    }
    public void processCamera(Mat mat, Callback<byte[]> callback) {
        nativeProcessCamera(mat.getNativeObjAddr(),callback);
    }

    public void stop(int type) {
        nativeStop(mNativeObj, type);
    }

    public void setMinFaceSize(int size) {
        nativeSetFaceSize(mNativeObj, size);
    }

    public void detect(Mat imageGray, MatOfRect faces) {
        nativeDetect(mNativeObj, imageGray.getNativeObjAddr(), faces.getNativeObjAddr());
    }

    public void release() {
        nativeDestroyObject(mNativeObj);
        mNativeObj = 0;
    }

    private long mNativeObj = 0;

    private static native long nativeCreateObject(String cascadeName, int minFaceSize);
    private static native void nativeDestroyObject(long thiz);
    private static native void nativeStart(String netPath, int RealWidth, int RealHeight, int width, int height, int type,  String fileName, Callback<byte[]> callback);
    public  static native void nativeStartCamera(String netPath, int RealWidth, int RealHeight, int width, int height);
    private static native void nativeProcessCamera(long inputImage, Callback<byte[]> callback);
    private static native void nativeStop(long thiz, int type);
    private static native void nativeSetFaceSize(long thiz, int size);
    private static native void nativeDetect(long thiz, long inputImage, long faces);

}
