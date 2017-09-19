#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include "ssd_detect.hpp"
#include "Detector_N_Tracker.hpp"
#include <glog/logging.h>

using namespace std;
using namespace caffe;

int main()
{
    string video_file("/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/ILSVRC2015_train_00755001.mp4");
    string label_file("/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/labelmap_voc.prototxt");
    string model_file("/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/deploy.prototxt");
    string weight_file("/Users/pitaloveu/WORKING_PROS/ssd_api/data_N_model/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel");
    const string & mean_value = "104,117,123";
    
    Detector_N_Tracker DNTracker( model_file , weight_file, label_file , mean_value );
    
    CHECK( DNTracker.init( video_file , 10 , 0.3 ));
    
    DNTracker.detectFrom( 0 );
    
//    cout << "shit not happen " << endl;
    DNTracker.writeToDisk( "/Users/pitaloveu/Desktop/test.avi");
    

      

  return 1;
}

