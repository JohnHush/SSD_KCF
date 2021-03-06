//#include <opencv2/core/core.hpp>
//#include <opencv2/video.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>

#include "ssd_detect.hpp"
#include "deepMAR.hpp"

using cv::Ptr;
using cv::VideoCapture;
using cv::Mat;
//using cv::Rect2d;
using cv::Scalar;

using std::string;
using std::cout;
using std::endl;

DEFINE_bool( USE_GPU , true , "use GPU or not " );
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

	if (argc < 8) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
		return 1;
	}

	const string& ssd_model  = argv[1];
	const string& ssd_weight = argv[2];
	const string& ssd_label  = argv[3];

	const string& mar_model  = argv[4];
	const string& mar_weight = argv[5];
	const string& mar_label  = argv[6];

	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const float confidence_threshold = FLAGS_confidence_threshold;
	const float scale = FLAGS_scale;
	const int skip = FLAGS_skip;
	const bool USE_GPU = FLAGS_USE_GPU;
	caffe::Caffe::Brew mode = USE_GPU ? caffe::Caffe::GPU : caffe::Caffe::CPU;

	const string& video_file = argv[7];

	VideoCapture cap( video_file );

	if( !cap.isOpened() )
		cout << "fail to open the video " << endl;

	Mat img;

	map<int , string> ssd_label_name;
	LabelMap ssd_label_map;
	ReadProtoFromTextFile( ssd_label , &ssd_label_map );
	CHECK( MapLabelToName( ssd_label_map , true , &ssd_label_name ) ) << "Duplicate labels";

	map<int , string> mar_label_name;
	LabelMap mar_label_map;
	ReadProtoFromTextFile( mar_label , &mar_label_map  );
	CHECK( MapLabelToName( mar_label_map , true, &mar_label_name ) ) << "Duplicate labels";

	Detector detector( ssd_model , ssd_weight , mean_file, mean_value , mode);
	MultiLabelClassifier classifier( mar_model , mar_weight ,	mean_file , mean_value , scale , mode );

	int video_width = cap.get( CV_CAP_PROP_FRAME_WIDTH );
	int video_heigh = cap.get( CV_CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter writer( "./default.avi" , CV_FOURCC('D','I','V','X') , 5 , cv::Size( video_width , video_heigh ) , true );
	int totalFrame = cap.get( CV_CAP_PROP_FRAME_COUNT );
	cap.set( CV_CAP_PROP_POS_FRAMES , 0 );
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
		cap.set( CV_CAP_PROP_POS_FRAMES , POS );

		Mat imgClone;
		img.copyTo( imgClone );

		timer.Start();
		std::vector<vector<float> > detections = detector.Detect(img);
		ssd_time += timer.MilliSeconds();

		std::vector<vector<float> > passD(0);
		std::vector<vector<float> > personD(0);
		std::vector<vector<float> > othersD(0);

		for( int iBBOX = 0 ; iBBOX < detections.size() ; ++ iBBOX )
		{
			if ( detections[iBBOX][2] >= confidence_threshold )
			{
				passD.push_back( detections[iBBOX] );
				if ( ssd_label_name[static_cast<int>(detections[iBBOX][1])] == "1" )
					personD.push_back( detections[iBBOX] );
				else
					othersD.push_back( detections[iBBOX] );
			}
		}

		for ( int i = 0 ; i < othersD.size() ; ++i )
		{
			const vector<float>& d = othersD[i];
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			CHECK_EQ(d.size(), 7);

			int Lbb = d[3] * img.cols;
			int Ubb = d[4] * img.rows;
			int Rbb = d[5] * img.cols;
			int Dbb = d[6] * img.rows;

			if ( Lbb >= img.cols || Rbb < 0 || Ubb >= img.rows || Dbb < 0 )
				continue;

			Lbb = std::max( 0 , Lbb );
			Ubb = std::max( 0 , Ubb );
			Rbb = std::min( img.cols -1 , Rbb );
			Dbb = std::min( img.rows -1 , Dbb );

			cv::Rect rect( Lbb , Ubb , Rbb-Lbb , Dbb - Ubb );

			if ( rect.width<=0 || rect.height <= 0 )
				continue;

			string label = ssd_label_name[static_cast<int>(d[1])];
			string score = std::to_string( d[2] );

			string show1 = label;
			string show2 = std::string( "            " ) + score;

			cv::rectangle( imgClone , rect , cv::Scalar( 0 , 255 , 0 ) , 2 );
			cv::putText( imgClone , show1 , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
			cv::putText( imgClone , show2 , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
		}

		std::vector<std::pair< const vector<float> , const cv::Rect > > personROI;
		std::vector<cv::Mat> imgVec;

		for (int i = 0; i < personD.size(); ++i)
		{
			const vector<float>& d = personD[i];
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			CHECK_EQ(d.size(), 7);

			int Lbb = d[3] * img.cols;
			int Ubb = d[4] * img.rows;
			int Rbb = d[5] * img.cols;
			int Dbb = d[6] * img.rows;

			if ( Lbb >= img.cols || Rbb < 0 || Ubb >= img.rows || Dbb < 0 )
				continue;

			Lbb = std::max( 0 , Lbb );
			Ubb = std::max( 0 , Ubb );
			Rbb = std::min( img.cols -1 , Rbb );
			Dbb = std::min( img.rows -1 , Dbb );

			cv::Rect rect( Lbb , Ubb , Rbb-Lbb , Dbb - Ubb );

			if ( rect.width<=0 || rect.height <= 0 )
				continue;

			personROI.push_back( std::make_pair( d , rect ));		

			cv::Mat img_deepMAR( img , rect );
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
//			string label = ssd_label_name[static_cast<int>(d[1])];

			const std::vector<float>& d = personROI[i].first;
			const cv::Rect& rect = personROI[i].second;
			string score = std::to_string( d[2] );

			cv::rectangle( imgClone , rect , cv::Scalar( 0 , 0 , 255 ) , 2 );
			cv::putText( imgClone , score , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,0,255) , 1 , 8 );

//			Mat img_deepMAR( img , rect );

//			std::vector<cv::Mat> img_deepMAR_Vec(1);
//			img_deepMAR_Vec[0] = img_deepMAR;

//			mar_time_count ++;
//			timer.Start();
//			std::vector<std::vector<int> > results_VEC = classifier.Analyze( img_deepMAR_Vec );
//			mar_time += timer.MilliSeconds();

			std::vector<int> results = results_Vec[i];

			int x_cor = rect.x;
			int y_cor = rect.y + 20;

			for ( int iattribute = 0 ; iattribute < results.size() ; ++ iattribute )
			{
				if ( results[iattribute] != 0 || iattribute == 0 )
				{
					y_cor += 30;

					string attribute = mar_label_name[ iattribute+1 ];
					if ( results[0] == 0 && iattribute == 0 )
						attribute = string("Male");

					cv::Rect rect_tmp( x_cor , y_cor-15 , 100 , 20 );
					if( x_cor < 0 || y_cor-15 < 0 || 
							x_cor+100 >= imgClone.size().width || y_cor-15 +20>=imgClone.size().height)
						continue;
					cv::Mat imgCloneROI = imgClone( rect_tmp );

					for( int irow =0 ; irow < imgCloneROI.rows ; ++ irow )
						for( int icol =0 ; icol < 3*imgCloneROI.cols ; ++ icol )
							imgCloneROI.at<uchar>(irow,icol) = uchar(imgCloneROI.at<uchar>(irow,icol )/3 + 32 );

					cv::putText( imgClone , attribute , cvPoint( x_cor , y_cor ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(255,255,255) , 1 , 8 );
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

