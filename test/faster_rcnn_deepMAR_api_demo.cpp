#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>
#include <sstream>

#include "api/api.hpp"
#include "deepMAR.hpp"
#include "caffe/FRCNN/util/frcnn_vis.hpp"

#include "opencv2/core/version.hpp"

using cv::VideoCapture;
using cv::Mat;
using cv::Scalar;

using std::string;
using std::cout;
using std::endl;
using std::stringstream;

DEFINE_int32( skip , 10 , "skip frame of the input video" );
DEFINE_string(mean_file, "",
		"The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
		"If specified, can be one value or can be same as image channels"
		" - would subtract from the corresponding channel). Separated by ','."
		"Either mean_file or mean_value should be provided, not both.");
DEFINE_double(confidence_threshold, 0.5,
		"Only store detections with score higher than the threshold.");
DEFINE_double(scale ,  0.00390625,
		"the scale factor when doing scale of the image");

DEFINE_string( model, "", "The model definition protocol buffer text file." );
DEFINE_string( weights, "", "Trained Model By Faster RCNN End-to-End Pipeline." );
DEFINE_string( default_c, "", "Default config file path." );

DEFINE_string( mar_model, "", "The model definition protocol buffer text file for MAR model" );
DEFINE_string( mar_weights, "", "Trained Model By deepMAR" );

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

	if (argc < 2) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
		return 1;
	}
#ifndef MacOS
  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();

  std::string mar_proto_file             = FLAGS_mar_model.c_str();
  std::string mar_model_file             = FLAGS_mar_weights.c_str();

  std::string output_dir = FLAGS_output_dir.c_str();

	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const float confidence_threshold = FLAGS_confidence_threshold;
	const float scale = FLAGS_scale;
	const int skip = FLAGS_skip;

	const string& video_file = argv[1];

	VideoCapture cap( video_file );

	if( !cap.isOpened() )
		cout << "fail to open the video " << endl;

	Mat img;

  API::Set_Config(default_config_file);
  API::Detector detector(proto_file, model_file);
#ifdef MacOS
  caffe::Caffe::Brew mode = caffe::Caffe::CPU;
#else
  caffe::Caffe::Brew mode = caffe::Caffe::GPU;
#endif
	MultiLabelClassifier classifier( mar_proto_file , mar_model_file ,	mean_file , mean_value , scale, mode );

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

	int video_width = cap.get( cap_frame_width_flag );
	int video_heigh = cap.get( cap_frame_height_flag );
  int total_frame = cap.get( cap_total_frame );

  std::cout << "total frame is " << total_frame << std::endl;

  stringstream file_name;

	cv::VideoWriter writer( "./default.avi" , cap_fourcc_flag , 5 , cv::Size( video_width , video_heigh ) , true );
	cap.set( cap_frame_prop_flag , 0 );
	int POS = 0;

	caffe::Timer timer;
	double ssd_time =0.;
	double mar_time =0.;
	int time_count =0;
	int mar_time_count =0;

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

		std::vector<caffe::Frcnn::BBox<float> > passD(0);
		std::vector<caffe::Frcnn::BBox<float> > personD(0);
		std::vector<caffe::Frcnn::BBox<float> > othersD(0);

		for( int iBBOX = 0 ; iBBOX < results.size() ; ++ iBBOX )
		{
			if ( results[iBBOX].confidence >= confidence_threshold )
			{
				passD.push_back( results[iBBOX] );
				if ( results[iBBOX].id == 15 )
					personD.push_back( results[iBBOX] );
				else
					othersD.push_back( results[iBBOX] );
			}
		}

		for ( int i = 0 ; i < othersD.size() ; ++i )
		{
			const caffe::Frcnn::BBox<float>& d = othersD[i];

			cv::Rect rect( d[0] , d[1] , d[2] - d[0] , d[3] - d[1] );

      std::string label = caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(), d.id);
      std::string score = std::to_string( d.confidence );

			string show1 = label;
			string show2 = std::string( "            " ) + score;

			cv::rectangle( imgClone , rect , cv::Scalar( 0 , 255 , 0 ) , 2 );
			cv::putText( imgClone , show1 , cv::Point( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cv::Scalar(0,255,0) , 1 , 8 );
			cv::putText( imgClone , show2 , cv::Point( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cv::Scalar(0,255,0) , 1 , 8 );
		}

		std::vector<std::pair< const caffe::Frcnn::BBox<float> , const cv::Rect > > personROI;
		std::vector<cv::Mat> imgVec;

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
        "_" << rect.width << "_" << rect.height << "_" << video_file <<
        "_" << video_width << "_" << video_heigh << "_" << personNum <<".jpg";

      std::cout << file_name.str() << std::endl;

      cv::imwrite( file_name.str() , img_deepMAR );

			imgVec.resize( i+1 );
			img_deepMAR.copyTo( imgVec[i] );
		}	

		mar_time_count ++;
		std::vector<std::vector<int> > results_Vec(0);

		timer.Start();
		if( imgVec.size() != 0 )
			results_Vec = classifier.Analyze( imgVec );
		mar_time += timer.MilliSeconds();

		for (int i = 0; i < personROI.size(); ++i)
		{
			const caffe::Frcnn::BBox<float>& d = personROI[i].first;
			const cv::Rect& rect = personROI[i].second;
			string score = std::to_string( d.confidence );

			cv::rectangle( imgClone , rect , cv::Scalar( 0 , 0 , 255 ) , 2 );
			cv::putText( imgClone , score , cv::Point( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cv::Scalar(0,0,255) , 1 , 8 );

			std::vector<int> att_results = results_Vec[i];

			int x_cor = rect.x;
			int y_cor = rect.y + 20;

			for ( int iattribute = 0 ; iattribute < att_results.size() ; ++ iattribute )
			{
        // we just show positive attributes or the gender
				if ( att_results[iattribute] != 0 || iattribute == 0 )
				{
					y_cor += 30;

					string attribute = caffe::Frcnn::GetClassName( caffe::Frcnn::LoadPA100Class() , iattribute );
					if ( att_results[0] == 0 && iattribute == 0 )
						attribute = string("Male");

          // make some background color
					cv::Rect rect_tmp( x_cor , y_cor-15 , 100 , 20 );
					if( x_cor < 0 || y_cor-15 < 0 || 
							x_cor+100 >= imgClone.size().width || y_cor-15 +20>=imgClone.size().height)
						continue;
					cv::Mat imgCloneROI = imgClone( rect_tmp );

					for( int irow =0 ; irow < imgCloneROI.rows ; ++ irow )
						for( int icol =0 ; icol < 3*imgCloneROI.cols ; ++ icol )
							imgCloneROI.at<uchar>(irow,icol) = uchar(imgCloneROI.at<uchar>(irow,icol )/3 + 32 );

					cv::putText( imgClone , attribute , cv::Point( x_cor , y_cor ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cv::Scalar(255,255,255) , 1 , 8 );
				}
			}
		}
		writer << imgClone;

    std::cout << POS << std::endl;
	//cv::namedWindow( "bbox show" );
  //      imshow( "bbox show" , imgClone );
  //     cvWaitKey(33);
	}
	LOG(INFO) << "ssd time = " << ssd_time/time_count << std::endl;
	LOG(INFO) << "mar time = " << mar_time/mar_time_count << std::endl;

	return 0;
}

