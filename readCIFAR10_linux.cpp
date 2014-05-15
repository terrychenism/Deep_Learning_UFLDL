// readCIFAR10.cpp

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <iostream>

#define ATD at<double>
#define elif else if

using namespace cv;
using namespace std;

void 
read_batch(char* filename, vector<Mat> &vec, Mat &label){
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels;
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.ATD(0, i) = (double)tplabel;
        }
    }
}

Mat 
concatenateMat(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_64FC1);
        Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_64FC1);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    divide(res, 255.0, res);
    return res;
}

Mat 
concatenateMatC(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width * 3, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_64FC3);
        vec[i].convertTo(img, CV_64FC3);
        vector<Mat> chs;
        split(img, chs);
        for(int j = 0; j < 3; j++){
            Mat ptmat = chs[j].reshape(0, height * width);
            Rect roi = cv::Rect(i, j * ptmat.rows, ptmat.cols, ptmat.rows);
            Mat subView = res(roi);
            ptmat.copyTo(subView);
        }
    }
    divide(res, 255.0, res);
    return res;
}

void
read_CIFAR10(Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){

    string filename;
    filename = "cifar-10-batches-bin/data_batch_1.bin";
    char *y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
   
    vector<Mat> batch1;
    Mat label1 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batch1, label1);
     delete[] y;

    filename = "cifar-10-batches-bin/data_batch_2.bin";
    y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
    
    vector<Mat> batch2;
    Mat label2 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batch2, label2);
    delete[] y;

    filename = "cifar-10-batches-bin/data_batch_3.bin";
    y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
    
    vector<Mat> batch3;
    Mat label3 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batch3, label3);
    delete[] y;

    filename = "cifar-10-batches-bin/data_batch_4.bin";
    y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
    
    vector<Mat> batch4;
    Mat label4 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batch4, label4);
    delete[] y;

    filename = "cifar-10-batches-bin/data_batch_5.bin";
  y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
    vector<Mat> batch5;
    Mat label5 = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batch5, label5);
    delete[] y;

    filename = "cifar-10-batches-bin/test_batch.bin";
      y = new char[filename.length() + 1]; 
    strcpy(y, filename.c_str());
    vector<Mat> batcht;
    Mat labelt = Mat::zeros(1, 10000, CV_64FC1);    
    read_batch(y, batcht, labelt);
    delete[] y;
    Mat mt1 = concatenateMat(batch1);
    Mat mt2 = concatenateMat(batch2);
    Mat mt3 = concatenateMat(batch3);
    Mat mt4 = concatenateMat(batch4);
    Mat mt5 = concatenateMat(batch5);
    Mat mtt = concatenateMat(batcht);

    Rect roi = cv::Rect(mt1.cols * 0, 0, mt1.cols, trainX.rows);
    Mat subView = trainX(roi);
    mt1.copyTo(subView);
    roi = cv::Rect(label1.cols * 0, 0, label1.cols, 1);
    subView = trainY(roi);
    label1.copyTo(subView);

    roi = cv::Rect(mt1.cols * 1, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt2.copyTo(subView);
    roi = cv::Rect(label1.cols * 1, 0, label1.cols, 1);
    subView = trainY(roi);
    label2.copyTo(subView);

    roi = cv::Rect(mt1.cols * 2, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt3.copyTo(subView);
    roi = cv::Rect(label1.cols * 2, 0, label1.cols, 1);
    subView = trainY(roi);
    label3.copyTo(subView);

    roi = cv::Rect(mt1.cols * 3, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt4.copyTo(subView);
    roi = cv::Rect(label1.cols * 3, 0, label1.cols, 1);
    subView = trainY(roi);
    label4.copyTo(subView);

    roi = cv::Rect(mt1.cols * 4, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt5.copyTo(subView);
    roi = cv::Rect(label1.cols * 4, 0, label1.cols, 1);
    subView = trainY(roi);
    label5.copyTo(subView);

    mtt.copyTo(testX);
    labelt.copyTo(testY);

}

int 
main()
{

    Mat trainX, testX;
    Mat trainY, testY;
    trainX = Mat::zeros(1024, 50000, CV_64FC1);  
    testX = Mat::zeros(1024, 10000, CV_64FC1);  
    trainY = Mat::zeros(1, 50000, CV_64FC1);  
    testX = Mat::zeros(1, 10000, CV_64FC1);  

    read_CIFAR10(trainX, testX, trainY, testY);
    cout<<"read success!"<<endl;
    

    return 0;
}
