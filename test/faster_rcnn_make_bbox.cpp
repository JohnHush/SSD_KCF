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


DEFINE_string( video_info , "" , "video info txt file, in a form:
                                    video_file_path  video_name" );
DEFINE_int32( skip , 10 , "skip frame of the input video" );
DEFINE_double(confidence_threshold, 0.5,
		"Only store detections with score higher than the threshold.");

DEFINE_string( model, "", "The model definition protocol buffer text file." );
DEFINE_string( weights, "", "Trained Model By Faster RCNN End-to-End Pipeline." );
DEFINE_string( default_c, "", "Default config file path." );

DEFINE_string( output_dir , "./img_dir" , "image dir to store detected images" );

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
  std::string fname( FLAGS_video_info );
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

	const float confidence_threshold = FLAGS_confidence_threshold;
	const int skip = FLAGS_skip;

  API::Set_Config(default_config_file);
  printf( "2\n" );
  API::Detector detector(proto_file, model_file);
#ifdef MacOS
  caffe::Caffe::Brew mode = caffe::Caffe::CPU;
#else
  caffe::Caffe::Brew mode = caffe::Caffe::GPU;
#endif

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

  stringstream file_name;

	caffe::Timer timer;

  for( std::map< std::string , std::string >::iterator it = PATH_SHOW_MAP.begin() ; 
      it != PATH_SHOW_MAP.end() ; ++ it )
  {
    VideoCapture cap( it->first );

	  double ssd_time =0.;
	  int time_count =0;
	  int POS = 190000;
	  int video_width = cap.get( cap_frame_width_flag );
	  int video_heigh = cap.get( cap_frame_height_flag );

    if( !cap.isOpened() )
      std::cout << "cannot open the video =>> " << it->second << std::endl;

    Mat img;

    while( cap.read(img) )
    {
      time_count ++;
      POS += skip;
      cap.set( cap_frame_prop_flag , POS );

      Mat imgClone;
      img.copyTo( imgClone );

      timer.Start();

      std::vector<caffe::Frcnn::BBox<float> > results;
      detector.predict( img ,  results );

      ssd_time += timer.MilliSeconds();

      std::vector<caffe::Frcnn::BBox<float> > personD(0);

      for( int iBBOX = 0 ; iBBOX < results.size() ; ++ iBBOX )
      {
        if ( results[iBBOX].confidence >= confidence_threshold )
        {
          if ( results[iBBOX].id == 15 )
            personD.push_back( results[iBBOX] );
        }
      }

      std::vector<std::pair< const caffe::Frcnn::BBox<float> , const cv::Rect > > personROI;

      int personNum = 0;
      for (int i = 0; i < personD.size(); ++i)
      {
        personNum ++;
        const caffe::Frcnn::BBox<float>& d = personD[i];

        // cv::Rect( x, y, width, height )
        cv::Rect rect( d[0] , d[1] , d[2] - d[0] , d[3] - d[1] );
        personROI.push_back( std::make_pair( d , rect ));

        cv::Mat img_deepMAR( img , rect );

        // write sub images into a user defined directory
        // the jpg name in a form frameID_x_y_w_h_.jpg
        file_name.str("");

        // write the cropped images respect to the bbox
        // the file name in a format:
        //     frameID_X_Y_W_H_cameraID_frameHeight_frameWidth_numIDinFrame.jpg
        file_name << output_dir << "/" <<  POS << "_" << rect.x << "_" << rect.y <<
          "_" << rect.width << "_" << rect.height << "_" << it->second <<
          "_" << video_width << "_" << video_heigh << "_" << personNum <<".jpg";

        std::cout << file_name.str() << std::endl;

        cv::imwrite( file_name.str() , img_deepMAR );
      }	

      std::cout << POS << std::endl;
    }
    LOG(INFO) << "ssd time = " << ssd_time/time_count << std::endl;
  }

  return 0;
}

