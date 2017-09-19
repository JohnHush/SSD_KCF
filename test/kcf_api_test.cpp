#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include "ssd_detect.hpp"

//using namespace cv;
using cv::Ptr;
using cv::TrackerKCF;
using cv::VideoCapture;
using cv::Mat;
using cv::Rect2d;
using cv::Scalar;

using namespace std;
using namespace caffe;

int main()
{
  Ptr<TrackerKCF> tracker = TrackerKCF::create();

  VideoCapture cap( "/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/ILSVRC2015_train_00755001.mp4" );

  if( !cap.isOpened() )
    cout << "fail to open the video " << endl;

  Mat frame;
  Rect2d roi;
  cap >> frame;

  roi.x =0;
  roi.y =0;
  roi.width = frame.cols/2;
  roi.height=frame.rows/2;

  rectangle( frame , roi , Scalar(255,0,0) , 2 , 2 );
//  imshow( "tracker" , frame );
//  waitKey(0);

    LabelMap label_map;
    ReadProtoFromTextFile( "/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/labelmap_voc.prototxt" , &label_map);
    
    map<int , string> label_to_name;
    if (!MapLabelToName(label_map, true, &label_to_name))
        cout << "Duplicate labels" << endl;
    
    for ( int index = 0 ; index < label_to_name.size() ; index ++ )
        cout << "label = " << index << "   name = " << label_to_name[index] << endl;
    
    
    const string & model_file = "/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/deploy.prototxt";
    const string & weights_file = "/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
    const string & mean_file = "";
    const string & mean_value = "104,117,123";
    const float confidence_threshold = 0.2;
    
    
    Detector detector(model_file, weights_file, mean_file, mean_value);
    
  int count =0;


    std::vector< cv::Rect > detection_purified;
  while( true )
  {
	if ( !cap.read(frame) )
	{
	    cout << "count = " << count << endl;
        break;
	    //return 0;
	}
      
      if ( 0 == count )
      {
          std::vector<vector<float> > detections = detector.Detect(frame);
          
          Mat imgClone = frame;
          
          for (int i = 0; i < detections.size(); ++i) {
          //for (int i = 0; i < 5; ++i) {
              const vector<float>& d = detections[i];
              // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
              CHECK_EQ(d.size(), 7);
              const float score = d[2];
              if (score >= confidence_threshold) {
                 // detection_purified.push_back( d );
                  
                                    cout << "" << " ";
                  cout << static_cast<int>(d[1]) << " ";
                  cout << score << " ";
                  cout << static_cast<int>(d[3] * frame.cols) << " ";
                  cout << static_cast<int>(d[4] * frame.rows) << " ";
                  cout << static_cast<int>(d[5] * frame.cols) << " ";
                  cout << static_cast<int>(d[6] * frame.rows) << std::endl;
                  
                  cv::Rect bbox;
                  bbox.x = d[3] * frame.cols;
                  bbox.y = d[4] * frame.rows;
                  bbox.width = d[5] * frame.cols - d[3] * frame.cols;
                  bbox.height = d[6] * frame.rows - d[4] * frame.rows;
                  
                 // vector<cv::Rect2d> rect_vec;
                  detection_purified.push_back( bbox );
                  
//                  string show_message = label_to_name[d[1]] ;//+ string( "   ") + to_string(d[2]);
//                  cv::rectangle( imgClone , bbox , cv::Scalar(255,0,0) , 2 );
//                  cv::putText( imgClone , show_message , cvPoint( bbox.x , bbox.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
              }
              
              
          }
//          imshow("tst" , imgClone);
 //         cv::waitKey(0);
      }//end of if( 0 == count)
      
//	imshow("tracker" , frame);
//	waitKey(100);
	count ++;
  }
    
    cap.set( CV_CAP_PROP_POS_FRAMES,0);
    
    cap.read(frame);
    
  //  Rect2d bbox( detection_purified[0][3] , detection_purified[0][4] , detection_purified[0][5] - detection_purified[0][3] ,
    //            detection_purified[0][6] - detection_purified[0][4] );
    
    
    Rect2d bbox = detection_purified[8];
//    cout << "bbox x = " << detection_purified[0].x;
//    cout << "bbox x = " << detection_purified[0].y;
//    cout << "bbox x = " << detection_purified[0].width;
//    cout << "bbox x = " << detection_purified[0].height;
//    
//    return 1;
    
 //   Rect2d bbox(10 , 10 ,55 , 55);
    tracker->init(frame , bbox);
    
    
    
//    cv::rectangle( frame , bbox , Scalar(0,0,255) , 2 , 1 );
//    imshow("ddd", frame );
//    cv::waitKey(0);
//    return 1;
    
    while( cap.read(frame) )
    {
        tracker->update(frame , bbox);
        cv::rectangle( frame , bbox , Scalar(0,0,255) , 2 , 1 );
        imshow( "Tracking" , frame );
        cv::waitKey(66);
    }
    
//    if (!cap.read(frame))
//        cout << "shit happen" <<endl;
//    imshow( "test", frame );
//    cv::waitKey(0);
  return 1;
}

