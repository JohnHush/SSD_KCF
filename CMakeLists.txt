cmake_minimum_required( VERSION 3.0 )

project( SSD_API C CXX )


include(CheckCXXCompilerFlag)  
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)  
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)  

if(COMPILER_SUPPORTS_CXX11)  
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")  
elseif(COMPILER_SUPPORTS_CXX0X)  
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")  
else()  
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")  
endif()


set( UNBUILDING_TRACKING ON )

include( /home/jh/working_lib/caffe/build/install/share/Caffe/CaffeConfig.cmake )
include( /usr/local/share/OpenCV/OpenCVConfig.cmake )

#set( OPENCVLIBS optimized /usr/local/lib/libopencv_tracking.dylib 
#		debug /usr/local/lib/libopencv_tracking.dylib )

add_subdirectory( src src_build )
add_subdirectory( test tst_build )

