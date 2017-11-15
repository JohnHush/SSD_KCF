#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepMAR.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
     "If specified, can be one value or can be same as image channels"
     " - would subtract from the corresponding channel). Separated by ','."
     "Either mean_file or mean_value should be provided, not both.");
DEFINE_double( scale , 0.00390625, 
		"scale factor of the image ");


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD by weiliu89\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file image_list  label_list\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;
	const float& scale = static_cast<float>(FLAGS_scale);

  // Initialize the network.
  MultiLabelClassifier classifier(model_file, weights_file , mean_file , mean_value ,scale);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ostream out(buf);

	const string& image_list = argv[3];
	const string& label_list = argv[4];

	std::ifstream images;
	std::ifstream labels;

	std::string image_name;
	std::string label_name;

	images.open( image_list.c_str() );
	labels.open( label_list.c_str() );

	int count = 0 ;
	int wrong_count = 0 ;

	int index = 0;

	vector<int> wrong_count_per_class(26,0);
	vector<float> accuracy_per_class(26,0.);

	while( std::getline( images , image_name ) )
	{
		cv::Mat img = cv::imread( image_name, -1);
		cv::Mat imgClone = img;
		CHECK(!img.empty()) << "Unable to decode image " << image_name;
		std::vector<int> results = classifier.Analyze( img );

		std::cout << std::endl;
		for ( int i = 0 ; i < results.size() ; i ++ )
			std::cout << results[i] << " ";
		std::cout << std::endl;

		std::getline( labels , label_name );
		std::stringstream ss;
		ss << label_name;
		int label;
		vector<int> int_label;
		while( ss >> label )
			int_label.push_back( label );

		for( int i = 0 ; i < 26 ;i ++ )
			std::cout << int_label[i] << " ";
		std::cout << std::endl;

		for ( int i = 0 ; i < 26 ; i ++ )
		{
			if ( int_label[i] != results[i] )
				wrong_count_per_class[i] ++;
		}

		count ++;
		if( count > 1000 )
			break;
	};

	for( int i = 0 ; i < 26 ; i ++ )
	{
		accuracy_per_class[i] = 1. -( wrong_count_per_class[i] /1000. );
		std::cout << "class = " << i << "  accuracy = " << accuracy_per_class[i] << std::endl;
	}

/*
*/
	return 0;
}
