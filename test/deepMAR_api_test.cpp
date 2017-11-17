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


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD by weiliu89\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
	const string& mean_file = FLAGS_mean_file;
	const string& mean_value = FLAGS_mean_value;

  // Initialize the network.
  MultiLabelClassifier classifier(model_file, weights_file , mean_file , mean_value ,0.3);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ostream out(buf);

	const string& file = argv[3];
	cv::Mat img = cv::imread(file, -1);
	cv::Mat imgClone = img;

	CHECK(!img.empty()) << "Unable to decode image " << file;

	std::vector<cv::Mat> imgVec;
	imgVec.push_back( img );

	std::vector<std::vector<int> > results_Vec = classifier.Analyze( imgVec );
	std::vector<int> results = results_Vec[0];

	for ( int i = 0 ; i < results.size() ; i ++ )
		std::cout << results[i] << " ";
/*
	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold) {
			out << file << " ";
			out << static_cast<int>(d[1]) << " ";
			out << score << " ";
			out << static_cast<int>(d[3] * img.cols) << " ";
			out << static_cast<int>(d[4] * img.rows) << " ";
			out << static_cast<int>(d[5] * img.cols) << " ";
			out << static_cast<int>(d[6] * img.rows) << std::endl;

			cv::Rect bbox;
			bbox.x = d[3] * img.cols;
			bbox.y = d[4] * img.rows;
			bbox.width = d[5] * img.cols - d[3] * img.cols;
			bbox.height = d[6] * img.rows - d[4] * img.rows;

			cv::rectangle( imgClone , bbox , cv::Scalar(255,0,0));
		}
	}
	cv::imshow( "bbox show" , img );
	cv::waitKey(0);
*/
	return 0;
}
