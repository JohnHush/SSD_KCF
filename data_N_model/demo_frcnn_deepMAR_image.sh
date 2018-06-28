#!/usr/bin/env sh

BUILD=build/tst_build/faster_rcnn_deepMAR_image_demo

$BUILD \
       --model data_N_model/deploy_faster_rcnn_vgg16_voc.prototxt \
       --weights data_N_model/VGG16_faster_rcnn_final.caffemodel \
       --default_c data_N_model/faster_rcnn_voc_config.json \
       --mar_model data_N_model/deploy_multi_label_classify.prototxt \
       --mar_weights data_N_model/PA100K_26_iter_10000.caffemodel \
      data_N_model/IMG_20180621_121055.jpg
