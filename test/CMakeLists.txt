
include_directories( AFTER "../src" )

file( GLOB_RECURSE tests ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

if( UNBUILDING_TRACKING )
  list( REMOVE_ITEM tests "${CMAKE_CURRENT_SOURCE_DIR}/kcf_api_test.cpp")
  list( REMOVE_ITEM tests "${CMAKE_CURRENT_SOURCE_DIR}/ssd_kcf_mixed.cpp")
endif()
if( True )
  list( REMOVE_ITEM tests "${CMAKE_CURRENT_SOURCE_DIR}/ssd_deepMAR_test.cpp")
endif()

foreach( source ${tests} )
	get_filename_component( name ${source} NAME_WE )
	add_executable( ${name} ${source} )

  if( APPLE )
    target_link_libraries( ${name} ssd_wrapper /Users/pitaloveu/working_libs/caffe/build/install/lib/libFRCNN_api.dylib ${Caffe_LIBRARIES} ${OPENCVLIBS})
  else()
    target_link_libraries( ${name} ssd_wrapper /home/jh/working_lib/caffe/build/install/lib/libFRCNN_api.so ${Caffe_LIBRARIES} ${OPENCVLIBS})
  endif()

	target_include_directories( ${name} PUBLIC ${Caffe_INCLUDE_DIRS} )
	target_compile_definitions( ${name} PUBLIC ${Caffe_DEFINITIONS} )
endforeach(source)
