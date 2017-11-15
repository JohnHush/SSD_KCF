#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>

#include "ssd_detect.hpp"
#include "deepMAR.hpp"

using cv::Ptr;
using cv::VideoCapture;
using cv::Mat;
using cv::Rect2d;
using cv::Scalar;

using std::string;
using std::cout;
using std::endl;

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

	const string& video_file = argv[7];

	VideoCapture cap( video_file );

	if( !cap.isOpened() )
		cout << "fail to open the video " << endl;

	Mat img;

//	cap.set( CV_CAP_PROP_POS_FRAMES,3400);

//	cap >> img;

	map<int , string> ssd_label_name;
	LabelMap ssd_label_map;
	ReadProtoFromTextFile( ssd_label , &ssd_label_map );
	CHECK( MapLabelToName( ssd_label_map , true , &ssd_label_name ) ) << "Duplicate labels";

	map<int , string> mar_label_name;
	LabelMap mar_label_map;
	ReadProtoFromTextFile( mar_label , &mar_label_map  );
	CHECK( MapLabelToName( mar_label_map , true, &mar_label_name ) ) << "Duplicate labels";

	Detector detector( ssd_model , ssd_weight , mean_file, mean_value );
	MultiLabelClassifier classifier( mar_model , mar_weight ,	mean_file , mean_value , scale );

	int video_width = cap.get( CV_CAP_PROP_FRAME_WIDTH );
	int video_heigh = cap.get( CV_CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter writer( "./default.avi" , CV_FOURCC('D','I','V','X') , 5 , cv::Size( video_width , video_heigh ) , true );

	int skip = 100;
	while( cap.read(img) )
	{
		for( int i = 0 ;i < skip ; ++i )
		{
			if ( !cap.read(img) )
				break;
		}

		Mat imgClone;
		img.copyTo( imgClone );

		std::vector<vector<float> > detections = detector.Detect(img);

		for (int i = 0; i < detections.size(); ++i)
		{
			const vector<float>& d = detections[i];
			// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
			CHECK_EQ(d.size(), 7);
			const float score = d[2];
			if (score >= confidence_threshold)
			{
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

				string label = ssd_label_name[static_cast<int>(d[1])];
				string score = std::to_string( d[2] );

				string show1 = label;
				string show2 = std::string( "            " ) + score;

				if ( label == "1" ) 
				{
					cv::rectangle( imgClone , rect , cv::Scalar( 0 , 0 , 255 ) , 2 );
					cv::putText( imgClone , score , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,0,255) , 1 , 8 );
				}

				else
				{	
					cv::rectangle( imgClone , rect , cv::Scalar( 0 , 255 , 0 ) , 2 );
					cv::putText( imgClone , show1 , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
					cv::putText( imgClone , show2 , cvPoint( rect.x , rect.y ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(0,255,0) , 1 , 8 );
				}

				if ( label == "1" )
				{
					Mat img_deepMAR( img , rect );

					std::vector<int> results = classifier.Analyze( img_deepMAR );

					int x_cor = rect.x;
					int y_cor = rect.y + 10;

					for ( int iattribute = 0 ; iattribute < results.size() ; ++ iattribute )
					{
						if ( results[iattribute] != 0 || iattribute == 0 )
						{
							y_cor += 20;

							string attribute = mar_label_name[ iattribute+1 ];
							if ( results[0] == 0 && iattribute == 0 )
								attribute = string("Male");
							cv::putText( imgClone , attribute , cvPoint( x_cor , y_cor ) , cv::FONT_HERSHEY_SIMPLEX , 0.5, cvScalar(255,0,0) , 1 , 8 );
						}
					}
				}
			}
		}
		writer << imgClone;

		cv::namedWindow( "bbox show" );
		cv::imshow( "bbox show" , imgClone );
		cv::waitKey(33);
	}
	return 0;
}

