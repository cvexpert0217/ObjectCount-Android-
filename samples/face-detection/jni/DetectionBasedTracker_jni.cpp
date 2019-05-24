#include <DetectionBasedTracker_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <time.h>

using namespace std;
using namespace cv;

typedef struct activity_context {
    JavaVM  *javaVM;
    jclass   mainActivityClz;
    jobject  mainActivityObj;
    pthread_mutex_t  lock;
    int      done;
} ActivityContext;
ActivityContext g_ctx;

jobject globalref;
static JavaVM* jvm = 0;
inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}


class PeopleCounter
{
public:
    //constructor for video detect
    PeopleCounter(cv::VideoCapture &cap,int captureFrameWidth, int captureFrameHeight, int width, int height,
                  std::string cnf_path, std::string wts_path, std::string nms_path,
                  float ct, float st, int iw, int ih, float zsf) :
            _capture(cap),
            _captureFrameWidth(captureFrameWidth),
            _captureFrameHeight(captureFrameHeight),
            _frameRegionToShow({ 0, 0, 0, 0 }),
            _frameRegionToShowPrevious({ 0, 0, 0, 0 }),
            _zoomSpeedFactor(zsf),
            _peopleQty(0),
            _modelConfigurationFile(cnf_path),
            _modelWeightsFile(wts_path),
            _classesFile(nms_path),
            _confThreshold(ct),
            _nmsThreshold(st),
            _inpWidth(iw),
            _inpHeight(ih),
            _screenWidth(width),
            _screenHeight(height),
            _threadsEnabled(true)
    {
        _frameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        _frameRegionToShowPrevious = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        _frameRegionToShowZoomed = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        // Setup the model
        _net = cv::dnn::readNetFromDarknet(_modelConfigurationFile, _modelWeightsFile);
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::ifstream ifs(_classesFile.c_str());
        std::string line;
        while (getline(ifs, line)) {
            _classes.push_back(line);
        }
    }

    //constructor for image and camera detect
    PeopleCounter(cv::Mat &image,int captureFrameWidth, int captureFrameHeight, int width, int height,
                  std::string cnf_path, std::string wts_path, std::string nms_path,
                  float ct, float st, int iw, int ih,  float zsf) :
            _image(image),
            _captureFrameWidth(captureFrameWidth),
            _captureFrameHeight(captureFrameHeight),
            _frameRegionToShow({ 0, 0, 0, 0 }),
            _frameRegionToShowPrevious({ 0, 0, 0, 0 }),
            _zoomSpeedFactor(zsf),
            _peopleQty(0),
            _modelConfigurationFile(cnf_path),
            _modelWeightsFile(wts_path),
            _classesFile(nms_path),
            _confThreshold(ct),
            _nmsThreshold(st),
            _inpWidth(iw),
            _inpHeight(ih),
            _screenWidth(width),
            _screenHeight(height),
            _threadsEnabled(true)
    {
        _frameRegionToShow = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        _frameRegionToShowPrevious = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        _frameRegionToShowZoomed = { 0, 0, _captureFrameWidth, _captureFrameHeight };
        // Setup the model
        _net = cv::dnn::readNetFromDarknet(_modelConfigurationFile, _modelWeightsFile);
        _net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        _net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::ifstream ifs(_classesFile.c_str());
        std::string line;
        while (getline(ifs, line)) {
            _classes.push_back(line);
        }
    }

    //set image to detect
    void setImage(cv::Mat& matImage )
    {
        _image = matImage;
    }

    //control thread status
    void setThreadEnabled(bool b)
    {
        _threadsEnabled = b;
    }

    //run threads to detect frames of video and call callback function java
    void runThreads(JNIEnv * jenv, void *completionCallback) {
        jvm->AttachCurrentThread(&jenv, NULL);

        jclass cbClass = jenv->FindClass("org/opencv/samples/facedetect/Callback");
        jmethodID method = jenv->GetMethodID(cbClass, "success", "(Ljava/lang/Object;)V");
        jbyteArray resultImage;
        resultImage = jenv->NewByteArray(4);
        std::thread producer_t(&PeopleCounter::producer, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::thread processor_t(&PeopleCounter::processor, this);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        while (_threadsEnabled) {
            usleep(1000);
            {
                std::lock(_mutexFrameRegion, _mutexFrameOverlay, _mutexFrameCapture);
                std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
                std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
                std::lock_guard<std::mutex> lckCapture(_mutexFrameCapture, std::adopt_lock);
                updateFrameRegionToShow();
                if (!_lastCapturedFrame.empty() && !_lastOverlayFrame.empty()) {
                    // Blur the background
                    cv::Mat blurred = cv::Mat::zeros(_lastCapturedFrame.size(),
                                                     _lastCapturedFrame.type());
                    cv::GaussianBlur(_lastCapturedFrame, blurred, cv::Size(15, 15), 0.0);

                    blurred.copyTo(_lastCapturedFrame, _blurMask);

                    cv::bitwise_or(_lastCapturedFrame, _lastOverlayFrame, _lastOverlayedFrame);
                }
                if (!_lastOverlayedFrame.empty()) {
                    cv::Mat frame = _lastOverlayedFrame(_frameRegionToShowZoomed);

                    padAspectRatio(frame, (float) _screenWidth / (float) _screenHeight);

                    cv::resize(frame, frame, cv::Size(_screenWidth, _screenHeight));

                    if (!frame.empty()) {
                        vector<cv::Mat> channels(3);
                        cv::split(frame, channels);
                        Mat alpha(channels[0].size(),CV_8UC1,Scalar(255));
                        Mat all_channel[4] = {channels[2],channels[1], channels[0] ,alpha}; //create an array of mat, here the channel order is BGRA
                        Mat dst; //ourput mat
                        merge(all_channel,4,dst); // merge to get the final result

                        if (resultImage == NULL || jenv->GetArrayLength( resultImage ) != dst.total()*4)
                            resultImage = jenv->NewByteArray(dst.total() * 4);
                        jbyteArray ret = matToByteArray(resultImage, jenv, dst);
                        jenv->CallVoidMethod(static_cast<jobject>(completionCallback), method, ret);
                    }
                }
            }
        }
        producer_t.join();
        processor_t.join();
    }

    //run detect image mat or video frame mat and call callback function in java
    void runDetectImage(JNIEnv * jenv, void *completionCallback) {
        jvm->AttachCurrentThread(&jenv, NULL);
        jclass cbClass = jenv->FindClass("org/opencv/samples/facedetect/Callback");
        jmethodID method = jenv->GetMethodID(cbClass, "success", "(Ljava/lang/Object;)V");
        jbyteArray resultImage;
        resultImage = jenv->NewByteArray(4);
        _lastCapturedFrame = _image.clone();
        processFrame(_image);
        std::lock(_mutexFrameRegion, _mutexFrameOverlay, _mutexFrameCapture);
        std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
        std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);
        std::lock_guard<std::mutex> lckCapture(_mutexFrameCapture, std::adopt_lock);
        updateFrameRegionToShow();
        if (!_lastCapturedFrame.empty() && !_lastOverlayFrame.empty()) {
            // Blur the background
            cv::Mat blurred = cv::Mat::zeros(_lastCapturedFrame.size(),
                                             _lastCapturedFrame.type());
            cv::GaussianBlur(_lastCapturedFrame, blurred, cv::Size(15, 15), 0.0);

            blurred.copyTo(_lastCapturedFrame, _blurMask);

            cv::bitwise_or(_lastCapturedFrame, _lastOverlayFrame, _lastOverlayedFrame);
        }
        if (!_lastOverlayedFrame.empty()) {
            cv::Mat frame = _lastOverlayedFrame(_frameRegionToShowZoomed);

            padAspectRatio(frame, (float) _screenWidth / (float) _screenHeight);

            cv::resize(frame, frame, cv::Size(_screenWidth, _screenHeight));

            if (!frame.empty()) {
                int frame_type = frame.type();
                vector<cv::Mat> channels(3);
                cv::split(frame, channels);
                Mat alpha(channels[0].size(),CV_8UC1,Scalar(255));
                Mat all_channel[4] = {channels[0],channels[1], channels[2] ,alpha}; //create an array of mat, here the channel order is BGRA
                Mat dst; //ourput mat
                merge(all_channel,4,dst); // merge to get the final result

                if (resultImage == NULL || jenv->GetArrayLength( resultImage ) != dst.total()*4)
                    resultImage = jenv->NewByteArray(dst.total() * 4);
                jbyteArray ret = matToByteArray(resultImage, jenv, dst);
                jenv->CallVoidMethod(static_cast<jobject>(completionCallback), method,
                                     ret);


            }
        }
    }

    int getPeopleQty() {
        return _peopleQty;
    }

private:
    // Get the names of the output layers
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net)
    {
        static std::vector<std::string> names;
        if (names.empty()) {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            //get the names of all the layers in the network
            std::vector<std::string> layersNames = net.getLayerNames();
            // Get the names of the output layers in names
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i) {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        return names;
    }

    // Filter out low confidence objects with non-maxima suppression
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
    {
        //Draw a rectangle displaying the bounding box
        rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        std::string label = cv::format("%.2f", conf);
        if (!_classes.empty()) {
            CV_Assert(classId < (int)_classes.size());
            label = _classes[classId] + ":" + label;
        }

        //Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = cv::max(top, labelSize.height);
        rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
    }


    int countPeople(cv::Mat& frame, const std::vector<cv::Mat>& outs)
    {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i) {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > _confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        // Perform non maximum suppression
        std::vector<int> indices;
        int peopleQty = 0;
        cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, indices);
        {
            std::lock(_mutexFrameRegion, _mutexFrameOverlay);
            std::lock_guard<std::mutex> lckRegion(_mutexFrameRegion, std::adopt_lock);
            std::lock_guard<std::mutex> lckOverlay(_mutexFrameOverlay, std::adopt_lock);

            _lastOverlayFrame = cv::Mat::zeros(frame.size(), frame.type());
            _blurMask = cv::Mat::ones(frame.size(), CV_8UC1);
            cv::Rect frameRegion = { _captureFrameWidth, _captureFrameHeight, (-1)*_captureFrameWidth, (-1)*_captureFrameHeight };

            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                cv::Rect box = boxes[idx];

                if (_classes[classIds[idx]] == "person") {
                    peopleQty++;
                    drawPred(classIds[idx], confidences[idx], box.x, box.y,
                             box.x + box.width, box.y + box.height, _lastOverlayFrame);

                    adjustBlurMask(box);

                    // Expand the frame region to show to contain all objects
                    adjustFrameRegion(frameRegion, box);
                }
            }
            if ((peopleQty > 0) && (frameRegion.height > 0 && frameRegion.width > 0)) {
                _frameRegionToShow = frameRegion;
            }
            else {
                _frameRegionToShow = cv::Rect(0, 0, _captureFrameWidth, _captureFrameHeight);
            }

            // Put efficiency information.
            // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            std::vector<double> layersTimes;
            double freq = cv::getTickFrequency() / 1000;
            double t = _net.getPerfProfile(layersTimes) / freq;
            std::string label = cv::format("Inference time for a frame : %.2f ms", t);
            cv::putText(_lastOverlayFrame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        return peopleQty;
    }
    string type2str(int type) {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }
    jbyteArray matToByteArray(jbyteArray resultImage, JNIEnv *env, const cv::Mat &image) {


        jbyte *_data = new jbyte[image.total() * 4];
        for (int i = 0; i < image.total() * 4; i++) {
            _data[i] = image.data[i];
        }
        env->SetByteArrayRegion(resultImage, 0, image.total() * 4, _data);
        delete[]_data;

        return resultImage;
    }

    void processFrame(cv::Mat& frame) {
        // Create a 4D blob from a frame.
        cv::Mat blob;

        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(_inpWidth, _inpHeight), cv::Scalar(0, 0, 0), true, false);
        // Nets forward pass
        std::vector<cv::Mat> outs;
        _net.setInput(blob);
        _net.forward(outs, getOutputsNames(_net));
        // Filter out low confidence objects
        _peopleQty = countPeople(frame, outs);
    }

    //zoom in and out frame region to show
    void updateFrameRegionToShow() {
        static int oldLeftTopX, oldLeftTopY, oldRightBottomX, oldRightBottomY;
        static int leftTopX, leftTopY, rightBottomX, rightBottomY;
        static int newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY;

        boxToPoints(_frameRegionToShowPrevious, oldLeftTopX, oldLeftTopY, oldRightBottomX, oldRightBottomY);
        boxToPoints(_frameRegionToShow, leftTopX, leftTopY, rightBottomX, rightBottomY);
        boxToPoints(_frameRegionToShowZoomed, newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY);

        newLeftTopX += std::ceil(_zoomSpeedFactor * (leftTopX - oldLeftTopX));
        newLeftTopY += std::ceil(_zoomSpeedFactor * (leftTopY - oldLeftTopY));
        newRightBottomX += std::ceil(_zoomSpeedFactor * (rightBottomX - oldRightBottomX));
        newRightBottomY += std::ceil(_zoomSpeedFactor * (rightBottomY - oldRightBottomY));

        pointsToBox(_frameRegionToShowZoomed, newLeftTopX, newLeftTopY, newRightBottomX, newRightBottomY);
        boundRegionToCaptureFrame(_frameRegionToShowZoomed);

        _frameRegionToShowPrevious = _frameRegionToShowZoomed;
    }

    void boundRegionToCaptureFrame(cv::Rect& region) {
        int leftTopX, leftTopY, rightBottomX, rightBottomY;

        boxToPoints(region, leftTopX, leftTopY, rightBottomX, rightBottomY);

        leftTopX = clip(leftTopX, 0, _captureFrameWidth);
        leftTopY = clip(leftTopY, 0, _captureFrameHeight);
        rightBottomX = clip(rightBottomX, 0, _captureFrameWidth);
        rightBottomY = clip(rightBottomY, 0, _captureFrameHeight);

        pointsToBox(region, leftTopX, leftTopY, rightBottomX, rightBottomY);
    }

    void adjustFrameRegion(cv::Rect& region, cv::Rect& box) {
        int leftTopX = std::min(box.x, region.x);
        int leftTopY = std::min(box.y, region.y);
        int rightBottomX = std::max(region.x + region.width, box.x + box.width);
        int rightBottomY = std::max(region.y + region.height, box.y + box.height);

        region.x = leftTopX;
        region.y = leftTopY;
        region.width = rightBottomX - leftTopX;
        region.height = rightBottomY - leftTopY;
    }

    void adjustBlurMask(cv::Rect& region) {
        boundRegionToCaptureFrame(region);
        _blurMask(region).setTo(cv::Scalar(0));
    }

    void padAspectRatio(cv::Mat& img, float ratio) {
        int width = img.cols;
        int height = img.rows;

        if (width > height) {
            int padding = (width / ratio - height) / 2;
            padding = clip(padding, 0, height);
            cv::copyMakeBorder(img, img, padding, padding, 0, 0, cv::BORDER_ISOLATED, 0);
        }
        else {
            int padding = (height * ratio - width) / 2;
            padding = clip(padding, 0, width);
            cv::copyMakeBorder(img, img, 0, 0, padding, padding, cv::BORDER_ISOLATED, 0);
        }
    }

    void boxToPoints(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY) {
        leftTopX = box.x;
        leftTopY = box.y;
        rightBottomX = box.x + box.width;
        rightBottomY = box.y + box.height;
    }

    void pointsToBox(cv::Rect& box, int& leftTopX, int& leftTopY, int& rightBottomX, int& rightBottomY) {
        box.x = leftTopX;
        box.y = leftTopY;
        box.width = rightBottomX - leftTopX;
        box.height = rightBottomY - leftTopY;
    }

    template <typename T>
    T clip(const T& n, const T& lower, const T& upper) {
        return std::max(lower, std::min(n, upper));
    }

    void producer() {
        cv::Mat frame;

        while (_threadsEnabled) {
            _capture.read(frame);
            {
                std::lock_guard<std::mutex> lck(_mutexFrameCapture);
                _lastCapturedFrame = frame.clone();
            }
            if (frame.empty()) {
                _threadsEnabled = false;
                break;
            }
        }
        if (_capture.isOpened()) {
            _capture.release();
        }
    }

    void processor() {
        cv::Mat frame;

        while (_threadsEnabled) {
            {
                std::lock_guard<std::mutex> lck(_mutexFrameCapture);
                frame = _lastCapturedFrame.clone();
            }

            if (!frame.empty()) {
                processFrame(frame);
            }
        }
    }

    cv::VideoCapture _capture;
    cv::Mat _image;
    cv::Mat _lastCapturedFrame;
    cv::Mat _lastOverlayFrame;
    cv::Mat _lastOverlayedFrame;
    cv::Mat _lastProcessedFrame;
    cv::Mat _blurMask;
    cv::Rect _frameRegionToShow;
    cv::Rect _frameRegionToShowPrevious;
    cv::Rect _frameRegionToShowZoomed;
    float _zoomSpeedFactor;
    int _captureFrameWidth;
    int _captureFrameHeight;
    int _peopleQty;

    cv::dnn::Net _net;                        // object detection neural network

    std::string _modelConfigurationFile; // network configuration file
    std::string _modelWeightsFile;        // network weights file
    std::string _classesFile;            // network classes file

    float _confThreshold;            // Confidence threshold
    float _nmsThreshold;            // Non-maximum suppression threshold
    int _inpWidth;                    // Width of network's input image
    int _inpHeight;                    // Height of network's input image
    int _screenWidth;
    int _screenHeight;
    std::vector<std::string> _classes;

    bool _threadsEnabled;
    std::mutex _mutexFrameCapture;
    std::mutex _mutexFrameRegion;
    std::mutex _mutexFrameOverlay;
};

JNIEXPORT void JNICALL
Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCallbacks(JNIEnv *env, jobject obj)
{

}

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
    {
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
    }

    virtual ~CascadeDetectorAdapter()
    {

    }

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};
PeopleCounter *peopleCounter;
struct DetectorAgregator
{
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;
    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter>& _mainDetector, cv::Ptr<CascadeDetectorAdapter>& _trackingDetector):
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector)
    {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

JNIEXPORT jlong JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject
        (JNIEnv * jenv, jclass, jstring jFileName, jint faceSize)
{
    return 0;
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject
        (JNIEnv * jenv, jclass, jlong thiz)
{
    try
    {
        if(thiz != 0)
        {
            ((DetectorAgregator*)thiz)->tracker->stop();
            delete (DetectorAgregator*)thiz;
        }
    }
    catch(const cv::Exception& e)
    {
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
}

//call from java when camera detect start for PeopleCounter constructor
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStartCamera
        (JNIEnv * jenv, jclass instance,jstring netPath,jint realWidth, jint realHeight,  jint width, jint height) {

    jenv->GetJavaVM(&jvm); //store jvm reference for later
    const char* jnetPath = jenv->GetStringUTFChars(netPath, NULL);
    string stdNetPath(jnetPath);

    cv::Mat image;

    peopleCounter = new PeopleCounter(image ,realWidth, realHeight,
                                      width, height,
                                      stdNetPath +"/net.cfg",
                                      stdNetPath +"/net.wts",
                                      stdNetPath +"/net.nms",
                                      0.5, 0.4,
                                      320, 320, 0.01);
}

//call from java to process camera frame detect
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeProcessCamera
        (JNIEnv * jenv, jclass instance, jlong image, jobject completionCallback)
{
    Mat mat = *((Mat*)image);
    peopleCounter->setImage(mat);
    peopleCounter->runDetectImage(jenv, completionCallback);
}

//call from java to detect image and video
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart
        (JNIEnv * jenv, jclass instance,jstring netPath,jint realWidth, jint realHeight, jint width, jint height, jint mode, jstring fileName, jobject completionCallback)
{
    int runMode = (int)mode;
    int videoCaptureWidth = (int)realWidth;
    int videoCaptureHeight = (int)realHeight;
    jenv->GetJavaVM(&jvm); //store jvm reference for later
    const char* jnamestr = jenv->GetStringUTFChars(fileName, NULL);
    string stdFileName(jnamestr);
    const char* jnetPath = jenv->GetStringUTFChars(netPath, NULL);
    string stdNetPath(jnetPath);
    cv::VideoCapture cap;
    cv::Mat image;
    std::cout << "step 2 - " << stdFileName << std::endl;
    if(mode == 1)
    {
        image = cv::imread(stdFileName, cv::IMREAD_COLOR);

        peopleCounter = new PeopleCounter(image,videoCaptureWidth, videoCaptureHeight,width, height,
                                          stdNetPath +"/net.cfg",
                                          stdNetPath +"/net.wts",
                                          stdNetPath +"/net.nms",
                                          0.5, 0.4,
                                          320, 320, 0.01);

        peopleCounter->runDetectImage(jenv, completionCallback);
    }
    else
    {
        bool b;
        if(mode == 0)
        {
            b = cap.open(stdFileName, cv::CAP_ANDROID);
        }
        else
        {
            b = cap.open(0);
        }
        peopleCounter = new PeopleCounter(cap, videoCaptureWidth, videoCaptureHeight,width, height,
                                          stdNetPath +"/net.cfg",
                                          stdNetPath +"/net.wts",
                                          stdNetPath +"/net.nms",
                                          0.5, 0.4,
                                          320, 320, 0.01);
        peopleCounter->runThreads(jenv, completionCallback);
    }
}

//call from java when detect stop for PeopleCounter desstructor
JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop
        (JNIEnv * jenv, jclass, jlong thiz, jint type)
{
    peopleCounter->setThreadEnabled(false);
    if ((int)type == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    delete peopleCounter;
}

JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize
        (JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    try
    {
        if (faceSize > 0)
        {
            ((DetectorAgregator*)thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(const cv::Exception& e)
    {
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
}


JNIEXPORT void JNICALL Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect
        (JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    try
    {
        vector<Rect> RectFaces;
        ((DetectorAgregator*)thiz)->tracker->process(*((Mat*)imageGray));
        ((DetectorAgregator*)thiz)->tracker->getObjects(RectFaces);
        *((Mat*)faces) = Mat(RectFaces, true);

    }
    catch(const cv::Exception& e)
    {
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
}
