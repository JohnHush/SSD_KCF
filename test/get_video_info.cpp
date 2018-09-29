#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <sstream>
#include <string>
#include <fstream>

#include "api/api.hpp"
#include "caffe/FRCNN/util/frcnn_vis.hpp"

#include "opencv2/core/version.hpp"

using cv::VideoCapture;
using cv::Mat;
using cv::Scalar;

using std::string;
using std::cout;
using std::endl;
using std::stringstream;

// parse all the video names
void parse_txt( std::string file_name , std::vector<std::string>& video_path)
{
  video_path.clear();
  std::fstream fs( file_name , std::fstream::in );

  std::string line;
  while( std::getline( fs , line ) )
    video_path.push_back( line );

  fs.close();
}

void split( std::istream* is , std::vector<std::string>& split_vec, 
    char delim )
{
  split_vec.clear();
  std::string tmp;
  while( std::getline( *is , tmp , delim ) )
    split_vec.push_back( tmp );
}


DEFINE_int32( skip , 10 , "skip frame of the input video" );
DEFINE_string(mean_file, "",
		"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
		"If specified, can be one value or can be same as image channels"
		" - would subtract from the corresponding channel). Separated by ','."
		"Either mean_file or mean_value should be provided, not both.");
DEFINE_double(confidence_threshold, 0.5,
		"Only store detections with score higher than the threshold.");

DEFINE_string( model, "", "The model definition protocol buffer text file." );
DEFINE_string( weights, "", "Trained Model By Faster RCNN End-to-End Pipeline." );
DEFINE_string( default_c, "", "Default config file path." );

DEFINE_string( output_dir , "./img_dir" , "image dir to store detected images" );
DEFINE_string( input_file , "" , "input video name " );

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;


#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Do detection using SSD and then DeepMAR \n"
			"Usage:\n"
			"    ssd_detect [FLAGS] ssd_model_file ssd_weights_file ssd_label deepMAR_model "
			"deepMAR_weights deepMAR_label VIDEO_FILE\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	//if (argc < 2) {
	//	gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
//		return 1;
//	}

#ifndef MacOS
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

  // get the Video Name Map
  std::string fname( "out_info.txt" );
  std::vector<std::string> video_path_name;
  parse_txt( fname , video_path_name );

  std::map< std::string , std::string > PATH_SHOW_MAP;
  
  for( std::vector<std::string>::iterator it = video_path_name.begin() ; 
      it != video_path_name.end() ; ++ it )
  {
    std::stringstream helper_ss;
    helper_ss.str( *it );

    std::vector<std::string> helper_split_vec;
    split( &helper_ss , helper_split_vec , ' ' );

    std::cout << *it << std::endl;

    std::cout << helper_split_vec.size() << std::endl;

    // change the synbol '/' in the split vec to '*'
    for( std::string::iterator it = helper_split_vec[1].begin() ;
        it != helper_split_vec[1].end() ; ++ it )
      if( *it == '/' )
        *it = '*';

    PATH_SHOW_MAP[ helper_split_vec[0] ] = helper_split_vec[1];
  }

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();

  std::string output_dir = FLAGS_output_dir.c_str();
  std::string input_file = FLAGS_input_file.c_str();

	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const float confidence_threshold = FLAGS_confidence_threshold;
	const int skip = FLAGS_skip;

#if CV_MAJOR_VERSION == 2
  int cap_frame_width_flag = CV_CAP_PROP_FRAME_WIDTH;
  int cap_frame_height_flag = CV_CAP_PROP_FRAME_HEIGHT;
  int cap_fourcc_flag = CV_FOURCC('D','I','V','X');
  int cap_frame_prop_flag = CV_CAP_PROP_POS_FRAMES;
  int cap_total_frame = CV_CAP_PROP_FRAME_COUNT;
#elif ( CV_MAJOR_VERSION == 3 || CV_MAJOR_VERSION == 4 )
  int cap_frame_width_flag = cv::CAP_PROP_FRAME_WIDTH;
  int cap_frame_height_flag = cv::CAP_PROP_FRAME_HEIGHT;
  int cap_fourcc_flag = cv::VideoWriter::fourcc('D','I','V','X');
  int cap_frame_prop_flag = cv::CAP_PROP_POS_FRAMES;
  int cap_total_frame = cv::CAP_PROP_FRAME_COUNT;
#endif

  VideoCapture cap( input_file );
  std::cout << cap.get( cap_total_frame ) << std::endl;

  return 0;
}

