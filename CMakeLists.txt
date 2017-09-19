cmake_minimum_required( VERSION 3.0 )

project( SSD_API C CXX )

include( /Users/pitaloveu/WORKING_LIB/ssd_caffe/build/install/share/Caffe/CaffeConfig.cmake )
#include( /usr/local/share/OpenCV/OpenCVConfig.cmake )

set( OPENCVLIBS optimized /usr/local/lib/libopencv_tracking.dylib 
		debug /usr/local/lib/libopencv_tracking.dylib )

add_subdirectory( src src_build )
add_subdirectory( test tst_build )

