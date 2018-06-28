#ifndef DEEP_MAR_HPP_
#define DEEP_MAR_HPP_

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

using namespace caffe;  // NOLINT(build/namespaces)

class MultiLabelClassifier{
	public:
		MultiLabelClassifier( const string& model_file,
													const string& weights_file,
			 										const string& mean_file,
													const string& mean_value,
													const float& scale_factor,
													caffe::Caffe::Brew mode = caffe::Caffe::CPU	);

		std::vector<std::vector<int> > Analyze( const std::vector<cv::Mat>& imgVec );
		std::vector<std::vector<int> > Analyze( const std::vector<cv::Mat>& imgVec,
                std::vector<std::vector<float> >& prob );

	protected:
		void SetMean(const string& mean_file, const string& mean_value);
		void WrapInputLayer( std::vector<cv::Mat>* input_channels );

		void Preprocess( const std::vector<cv::Mat>& imgVec, std::vector<cv::Mat>* input_channels );
	
	private:
		shared_ptr<Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		int num_batch_;
		cv::Mat mean_;
		float scale_factor_;
};

#endif
