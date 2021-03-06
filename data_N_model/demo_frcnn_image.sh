#!/usr/bin/env sh

BUILD=/home/jh/working_pros/ssd_kcf/build/tst_build/faster_rcnn_api_demo

$BUILD \
       --model deploy_faster_rcnn_vgg16_voc.prototxt \
       --weights VGG16_faster_rcnn_final.caffemodel \
       --default_c faster_rcnn_voc_config.json \
     fish-bike.jpg 
