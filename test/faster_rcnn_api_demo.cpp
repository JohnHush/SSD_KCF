#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "api/api.hpp"
#include "caffe/FRCNN/util/frcnn_vis.hpp"

DEFINE_string(model, "", "The model definition protocol buffer text file.");

DEFINE_string(weights, "", "Trained Model By Faster RCNN End-to-End Pipeline.");

DEFINE_string(default_c, "", "Default config file path.");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Detecting using faster RCNN C++ version\n"
        "The Python version is written by Ross. GirShirck\n"
        "The Matlab version is written by Shaoqing Ren, and also the original paper writer\n"
        "Usage:\n"
        "    faster_rcnn_api_demo [FLAGS] img_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "too little arguments");
    return 1;
  }

  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();

  std::string img_file               = argv[1];
  cv::Mat img = cv::imread( img_file , -1 );

  CHECK(!img.empty()) << "Unable to decode image " << img_file;

  API::Set_Config(default_config_file);
  API::Detector detector(proto_file, model_file);

  caffe::Timer time_;

  std::vector<caffe::Frcnn::BBox<float> > results;
  time_.Start();
  detector.predict( img ,  results );

  LOG(INFO) << "Predicting cost " << time_.MilliSeconds() << " ms.";
  LOG(INFO) << "There are " << results.size() << " objects in picture.";

  for (size_t obj = 0; obj < results.size(); obj++)
    LOG(INFO) << results[obj].to_string();

  for (int label = 0; label < caffe::Frcnn::FrcnnParam::n_classes; label++) {
    std::vector<caffe::Frcnn::BBox<float> > cur_res;
    for (size_t idx = 0; idx < results.size(); idx++) {
      if (results[idx].id == label) {
        cur_res.push_back( results[idx] );
      }
    }
    if (cur_res.size() == 0) continue;

    cv::Mat ori ;
    img.convertTo(ori, CV_32FC3);
    caffe::Frcnn::vis_detections(ori, cur_res, caffe::Frcnn::LoadVocClass() );

    std::string name = img_file;
    char xx[100];
    sprintf(xx, "%s_%s.jpg", name.c_str(), caffe::Frcnn::GetClassName(caffe::Frcnn::LoadVocClass(),label).c_str());

    cv::imwrite(std::string(xx), ori);
  }

  return 0;
}
