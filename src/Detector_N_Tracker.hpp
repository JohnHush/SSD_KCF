//
//  Detector_N_Tracker.hpp
//  SSD_API
//
//  Created by John Hush on 18/09/2017.
//
//

#ifndef Detector_N_Tracker_hpp
#define Detector_N_Tracker_hpp

#include <stdio.h>
#include "ssd_detect.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

class Detector_N_Tracker
{
public:
    class BBOX
    {
    public:
        string image_id;
        int label;
        float score;
        int x;
        int y;
        int width;
        int heigh;
        //[image_id , label , score , x , y , width , height ]
        
        BBOX( const string& _image_id , const int& _label , const float& _score ,
             const int& _x, const int& _y ,const int& _width , const int& _heigh)
        :image_id(_image_id),label(_label),score(_score),x(_x),y(_y),width(_width),heigh(_heigh)
        {}
        BBOX(){}
    };
protected:
    Detector detector_;
    cv::Ptr<cv::TrackerKCF> tracker_;
    cv::VideoCapture cap_;
    vector<vector<BBOX> > bbox_;
    map<int , string> label_to_name_;
    cv::Mat frame_;

    string video_name_;
    int frame_width_;
    int frame_heith_;
    int frame_number_;
    int detection_gap_;
    float detection_threshold_;
    int start_frame_;
    int end_frame_;
    float resize_ratio_;
    bool ifResize_;
    cv::Size resize_size_;
    
    string model_file_;
    string weights_file_;
    string label_file_;
    string mean_value_;
    
public:
    Detector_N_Tracker( const string& model_file , const string& weights_file,
                       const string& label_file , const string& mean_value )
    :detector_(Detector( model_file , weights_file , "" , mean_value ))
    {
        model_file_ = model_file;
        weights_file_ = weights_file;
        label_file_ = label_file;
        mean_value_ = mean_value;
    }
    
    void init( const string& video_name , const int& detection_gap , const float& detection_threshold , const float& resize_ratio = 1. );
    
    void detectFromTo( const int& start , const int& end );
    void detectFrom( const int& start ){
        detectFromTo( start , frame_number_ );
    }
    void detectTo( const int& end ){
        detectFromTo( 0 , end );
    }
    void detect(){
        detectFromTo( 0 , frame_number_ );
    }
    void writeToDisk( const string& address , const bool& withLabel = true , const bool& withScore = false );
    
};

#endif /* Detector_N_Tracker_hpp */
