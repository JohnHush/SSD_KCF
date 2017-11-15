
include_directories( AFTER "../src" )

file( GLOB_RECURSE tests ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

if( UNBUILDING_TRACKING )
  list( REMOVE_ITEM tests "${CMAKE_CURRENT_SOURCE_DIR}/kcf_api_test.cpp")
  list( REMOVE_ITEM tests "${CMAKE_CURRENT_SOURCE_DIR}/ssd_kcf_mixed.cpp")
endif()

foreach( source ${tests} )
	get_filename_component( name ${source} NAME_WE )
	add_executable( ${name} ${source} )
	target_link_libraries( ${name} ssd_wrapper ${Caffe_LIBRARIES} ${OPENCVLIBS})
	target_include_directories( ${name} PUBLIC ${Caffe_INCLUDE_DIRS} )
	target_compile_definitions( ${name} PUBLIC ${Caffe_DEFINITIONS} )
endforeach(source)
