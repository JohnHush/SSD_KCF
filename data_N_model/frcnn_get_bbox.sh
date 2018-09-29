#!/usr/bin/env sh

BUILD=/home/jh/working_pros/ssd_kcf/build/tst_build/faster_rcnn_make_bbox

$BUILD \
       --model deploy_faster_rcnn_vgg16_voc.prototxt \
       --weights VGG16_faster_rcnn_final.caffemodel \
       --default_c faster_rcnn_voc_config.json \
       --skip 1 \
       --gpu_id 1\
       --video_info test1.txt
