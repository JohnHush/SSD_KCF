//
//  Detector_N_Tracker.cpp
//  SSD_API
//
//  Created by John Hush on 18/09/2017.
//
//

#include "Detector_N_Tracker.hpp"
#include <glog/logging.h>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <ctime>

void Detector_N_Tracker::init( const string& video_name , const int& detection_gap ,
                            const float& detection_threshold , const float& resize_ratio)
{
    using namespace std;
    video_name_ = video_name;
    cap_ = cv::VideoCapture( video_name );
    
    CHECK( cap_.isOpened() ) << "Cann't Open the fucking video:  " << video_name ;
    
    frame_number_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    
    LabelMap label_map;
    
    ReadProtoFromTextFile( label_file_ , &label_map);
    CHECK( MapLabelToName(label_map, true, &label_to_name_) ) << "Duplicate labels";
    
    detection_gap_ = detection_gap;
    detection_threshold_ = detection_threshold;
    
    bbox_.resize( frame_number_ );
    
    cv::Mat frame_in;
    CHECK(cap_.read(frame_in)) << "cannot open the video when initing";
    
    if ( fabs( resize_ratio - 1. ) > 1.E-5 )
    {
        cv::Size resize_size( frame_in.cols * resize_ratio , frame_in.rows * resize_ratio );
        ifResize_ = true;
        resize_size_ = resize_size;
        cv::resize( frame_in , frame_ , resize_size );
    }
    else
    {
        frame_ = frame_in;
        ifResize_ = false;
        resize_size_ = cv::Size( frame_in.cols , frame_in.rows );
    }
    
    resize_ratio_ = resize_ratio;
    frame_width_ = frame_.cols;
    frame_heith_ = frame_.rows;

}

void Detector_N_Tracker::detectFromTo( const int& start , const int& end )
{
    CHECK_GE( start , 0 ) << "start index out of range ";
    CHECK_LE( end , frame_number_) << "end index out of range!";
    CHECK_GT( end , start) << "start and end index are incompatible!";
    
    start_frame_ = start;
    end_frame_ = end;

    cv::Mat frame_in;
    cap_.set( CV_CAP_PROP_POS_FRAMES, start );
    CHECK( cap_.read(frame_in) ) << "CANNOT read the original frame from the video data ";
    bbox_.clear();
    bbox_.resize( end - start );
    // the bbox_ contain every bounding box in every frame;
    
    const int detection_time = (end-start)/detection_gap_ + 1;
    
    clock_t t1,t2,t3;
    
    for( int Dindex = 0 ; Dindex < detection_time ; ++Dindex )
    {
        t1 = clock();
        
        cap_.set(  CV_CAP_PROP_POS_FRAMES, start + Dindex * detection_gap_ );
        CHECK( cap_.read(frame_in)) << "cannot read the original frame";
        
        if ( ifResize_ )
            cv::resize( frame_in , frame_ , resize_size_ );
        else
            frame_ = frame_in;
        
       // cv::Mat frameClone = frame_;
        //vector<vector<float> > detections = detector_.Detect(frameClone);
        vector<vector<float> > detections = detector_.Detect(frame_);
        vector<BBOX> detection_purified;
        
        // detection purified format: [image_id , label , score , x , y , width , height ].
        for (int i = 0; i < detections.size(); ++i)
        {
            const vector<float>& d = detections[i];
            // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            CHECK_EQ(d.size(), 7);
            const float score = d[2];
            if ( score >= detection_threshold_ )
            {
                
                BBOX detect_tmp;
                
                detect_tmp.image_id = std::to_string(d[0]);
                detect_tmp.label    = int(d[1]);
                detect_tmp.score    = d[2];
                detect_tmp.x        = int(d[3] * frame_.cols );
                detect_tmp.y        = int(d[4] * frame_.rows );
                detect_tmp.width    = int( d[5] * frame_.cols - d[3] * frame_.cols );
                detect_tmp.heigh    = int( d[6] * frame_.rows - d[4] * frame_.rows );
                
                detection_purified.push_back( detect_tmp);
            }
        }// end of detection.size LOOP
        // detecting every possible bounding box in one frame with an invertal.
        // the capability of the variaty of detection depends on the detection model
        // which was originally trained by Liu Wei,2016. SSD arxiv paper.
        
        t2 = clock();
        
        for ( int i = 0  ; i < detection_purified.size() ; ++ i )
        {
            cap_.set(  CV_CAP_PROP_POS_FRAMES, start + Dindex * detection_gap_ );
            CHECK( cap_.read(frame_in) );
            if ( ifResize_ )
                cv::resize( frame_in , frame_ , resize_size_ );
            else
                frame_ = frame_in;
            
            cv::Rect2d roi( detection_purified[i].x , detection_purified[i].y ,
                           detection_purified[i].width , detection_purified[i].heigh );
            
            bbox_[Dindex * detection_gap_].push_back( detection_purified[i] );
            
            tracker_ = cv::TrackerKCF::create();
            tracker_->init( frame_ , roi);
            
            for( int Tindex = 0 ; Tindex < detection_gap_ -1 ; Tindex ++ )
            {
                if ( !cap_.read(frame_in) )
                    break;
                
                if ( ifResize_ )
                    cv::resize( frame_in , frame_ , resize_size_ );
                else
                    frame_ = frame_in;
                
                tracker_->update(frame_ , roi );
                
                BBOX bbox_tmp(detection_purified[i].image_id , detection_purified[i].label ,
                              detection_purified[i].score , roi.x , roi.y , roi.width , roi.height );
                
                bbox_[Dindex * detection_gap_ + Tindex +1 ].push_back( bbox_tmp );
                
            }
        }
        
        t3 = clock();
        std::cout << "Detecting time = " << float(t2-t1)/CLOCKS_PER_SEC<< "s" << std::endl;
        std::cout << "Tracking time = " << float(t3-t2)/CLOCKS_PER_SEC<<"s" << std::endl << std::endl;
        // tracking every bbox in the original frame, tracking with a length detection_gap_;
    }//end of Dindex LOOP
    
}

void Detector_N_Tracker::writeToDisk( const string& address , const bool& withLabel , const bool& withScore )
{
    cv::VideoWriter writer( address , CV_FOURCC('D','I','V','X') , 5 ,
                           cv::Size(frame_width_ , frame_heith_) , true);
    
    cap_.set( CV_CAP_PROP_POS_FRAMES , 0 );
    int frame_count = 0;
    cv::Mat frame_in;
    
    while( cap_.read(frame_in) )
    {
        //cv::Mat frameClone = frame_;
        if ( ifResize_ )
            cv::resize( frame_in , frame_ , resize_size_ );
        else
            frame_ = frame_in;
        
        if( frame_count >= start_frame_ && frame_count < end_frame_ )
        {
            int frame_index = frame_count - start_frame_;
            for ( int Bindex = 0 ; Bindex < bbox_[frame_index].size() ; ++ Bindex )
            {
                cv::Rect rect( bbox_[frame_index][Bindex].x ,
                               bbox_[frame_index][Bindex].y ,
                               bbox_[frame_index][Bindex].width ,
                               bbox_[frame_index][Bindex].heigh );
                
                string score = std::to_string(bbox_[frame_index][Bindex].score);
                string label_name = label_to_name_[ bbox_[frame_index][Bindex].label];
                
                if ( (frame_count - start_frame_ )%detection_gap_ != 0 )
                    cv::rectangle( frame_ , rect , cv::Scalar(255,0,0) , 2 );
                else
                    cv::rectangle( frame_ , rect , cv::Scalar(0,0,255) , 2 );
                // if this frame is detection frame, then decorate it with different color
                // and longer duration.
                
                if ( withLabel )
                {
                    string show_message = label_name;
                    cv::putText( frame_ , show_message , cvPoint( rect.x , rect.y ) ,
                                cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
                }
                if ( withScore )
                {
                    string show_message = string("    ") + score ;
                    cv::putText( frame_ , show_message , cvPoint( rect.x , rect.y ) ,
                                cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
                }
            }
        }// if frame in the range of tracking frame
        // we need to add the annotation boxes into the frame
        
//        cv::imshow( "test" , frame_ );
//        cv::waitKey(50);
        writer << frame_;
        if ( (frame_count - start_frame_ )%detection_gap_ == 0 )
            for( int icolor = 0 ; icolor < 10 ; icolor ++ )
                writer << frame_;
        // write every frame into the new created video file but with added bboxes.
        
        frame_count ++;
    }

}





