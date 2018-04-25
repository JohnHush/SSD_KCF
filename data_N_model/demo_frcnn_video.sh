#!/usr/bin/env sh

BUILD=/home/jh/working_pros/ssd_kcf/build/tst_build/faster_rcnn_deepMAR_api_demo

$BUILD \
       --model deploy_faster_rcnn_vgg16_voc.prototxt \
       --weights VGG16_faster_rcnn_final.caffemodel \
       --default_c faster_rcnn_voc_config.json \
       --mar_model deploy_multi_label_classify.prototxt \
       --mar_weights PA100K_26_iter_10000.caffemodel \
       --skip 1 \
       ILSVRC2015_train_00755001.mp4
